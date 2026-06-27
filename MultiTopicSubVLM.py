import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
import os
import sys
import json
from io import BytesIO
from traceback import format_exc
from PIL import Image
import base64
from typing import List, Optional
from pydantic import BaseModel

from std_msgs.msg import String

# Import EntityArray for entity tracking and Skeleton2DArray for body keypoints
from hri_msgs.msg import EntityArray, Skeleton2DArray

from src.vlm_client import VLMClient

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroqAPI.env"),
    override=True
)

# Mapping from Skeleton2D keypoint index to human-readable name (OpenPose COCO convention)
KEYPOINT_NAMES = {
    0: "NOSE", 1: "NECK",
    2: "RIGHT_SHOULDER", 3: "RIGHT_ELBOW", 4: "RIGHT_WRIST",
    5: "LEFT_SHOULDER", 6: "LEFT_ELBOW", 7: "LEFT_WRIST",
    8: "RIGHT_HIP", 9: "RIGHT_KNEE", 10: "RIGHT_ANKLE",
    11: "LEFT_HIP", 12: "LEFT_KNEE", 13: "LEFT_ANKLE",
    14: "LEFT_EYE", 15: "RIGHT_EYE",
    16: "LEFT_EAR", 17: "RIGHT_EAR",
}

# ---------------------------------------------------------------------------
# Tool definition for Groq function/tool calling.
#
# Strategy: instead of asking the model to output JSON in the message content,
# we define a "tool" whose parameters ARE the scene graph schema. Setting
# tool_choice to force this specific function means the model MUST populate
# the arguments with valid JSON — completely bypassing <think> blocks,
# markdown fences, and unsupported response_format modes.
#
# The Pydantic classes below are kept for reference/validation; the actual
# enforcement happens via SCENE_GRAPH_TOOL passed to the Groq API.
# ---------------------------------------------------------------------------

class SpatialInfo(BaseModel):
    box_2d: List[int]               # [ymin, xmin, ymax, xmax] absolute pixels

class EntityNode(BaseModel):
    id: int
    label: str
    type: str                       # "object" | "human" | "structural"
    states: List[str]
    spatial_info: SpatialInfo
    action_description: Optional[str]

class Relationship(BaseModel):
    subject_id: int
    predicate: str
    object_id: int

class SceneGraphSchema(BaseModel):
    entities: List[EntityNode]
    relationships: List[Relationship]

# Groq tool definition — JSON Schema format understood by the chat/completions API.
# Kept deliberately flat: Groq tool calling does not support JSON Schema union types
# (e.g. ["string","null"]), array size constraints (minItems/maxItems), or nested
# $ref. All constraints are expressed in field descriptions instead.
SCENE_GRAPH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "create_scene_graph",
            "description": (
                "Record the complete scene graph extracted from the camera frame and "
                "sensor data. Call this tool exactly once with ALL detected entities "
                "and ALL spatial relationships."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "All entities detected in the scene.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "integer",
                                    "description": "Tracker track_id. For untracked structural elements use IDs >= 1000."
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Snake_case name of the entity, e.g. coffee_mug, kitchen_counter."
                                },
                                "entity_type": {
                                    "type": "string",
                                    "description": "Category: 'object' for movable items, 'human' for people, 'structural' for fixed env elements."
                                },
                                "states": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "One or more applicable states. Valid values: "
                                        "open, closed, empty, full, dirty, clean, hot, cold, "
                                        "turned_on, turned_off, stable, unstable, broken, "
                                        "standing, sitting, walking, reaching, looking_at, "
                                        "interacting, neutral, gesturing, reachable, occluded, "
                                        "held_by, static, moving, unknown."
                                    )
                                },
                                "box_2d": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "Bounding box as [ymin, xmin, ymax, xmax] in absolute pixel integers."
                                },
                                "action_description": {
                                    "type": "string",
                                    "description": "For humans: short verb phrase (<=8 words) describing the current action. For objects/structural: use empty string or 'none'."
                                }
                            },
                            "required": ["id", "label", "entity_type", "states", "box_2d", "action_description"]
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "description": "All spatial and semantic relationships between entities.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject_id": {
                                    "type": "integer",
                                    "description": "Entity id of the subject."
                                },
                                "predicate": {
                                    "type": "string",
                                    "description": (
                                        "Relationship type. Valid values: "
                                        "on_top_of, inside, part_of, touching, not_touching, "
                                        "embedded_in, next_to, near, above, below, in_front_of, "
                                        "behind, on_the_left_of, on_the_right_of, facing, "
                                        "occluding, holding, held_by, pointed_by, looking_at, "
                                        "operating, interacting."
                                    )
                                },
                                "object_id": {
                                    "type": "integer",
                                    "description": "Entity id of the object."
                                }
                            },
                            "required": ["subject_id", "predicate", "object_id"]
                        }
                    }
                },
                "required": ["entities", "relationships"]
            }
        }
    }
]


# Force the model to always call this specific tool (no free-form text output).
SCENE_GRAPH_TOOL_CHOICE = {"type": "function", "function": {"name": "create_scene_graph"}}

# ---------------------------------------------------------------------------

class MultiTopicListener(Node):
    # This listener subscribes to the compressed image, entity detection, human detection,
    # and skeleton keypoint topics. It synchronizes them and sends merged data to the VLM
    # to generate a detailed Scene Graph with relationships and states.
    def __init__(self):
        super().__init__("multi_listener")
        self.counter_ = 0
        self.Analyzing = False

        self.image_sub = Subscriber(
            self,
            CompressedImage,
            "/camera/image_raw/compressed"
        )
        self.entity_sub = Subscriber(
            self,
            EntityArray,
            "/entities/detected"
        )
        self.human_sub = Subscriber(
            self,
            EntityArray,
            "/humans/detected"
        )
        self.skeleton_sub = Subscriber(
            self,
            Skeleton2DArray,
            "/humans/bodies/detected"
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.entity_sub, self.human_sub, self.skeleton_sub],
            queue_size=10,
            slop=0.5
        )
        self.sync.registerCallback(self.synchronized_callback)
        self.get_logger().info("Subscribed and synchronized image, entity, human & skeleton topics.")

        # Create a publisher to send the Scene Graph to the LLM Decision Maker                                                                                                   
        self.scene_graph_pub = self.create_publisher(                                                                                                                            
            String,                                                                                                                                                              
            '/scene_graph',                                                                                                                                                      
            10                                                                                                                                                                   
        )     

        # Configure and Initialize VLM Client
        self.use_nebius = False
        self.use_groq = True

        if self.use_nebius:
            model_parameters = self.test_nebius_vlm()
        elif self.use_groq:
            model_parameters = self.test_groq_vlm()
        else:
            raise ValueError("No VLM provider selected")

        self.get_logger().info(f"Initializing VLMClient with model: {model_parameters['model_name']}")
        self.vlm = VLMClient(**model_parameters)

    def test_groq_vlm(self):
        return {
            "model_name": "groq/qwen3.6-27b",
            'temperature': 0.0,
            'max_tokens': 1500,
            'top_p': 1.0
        }

    def test_nebius_vlm(self):
        return {
            "model_name": "nebius/qwen3-2.5-70b",
            'temperature': 0.0,
            'max_tokens': 1500,
            'top_p': 1.0,
        }

    def synchronized_callback(self, image_msg, entity_msg, human_msg, skeleton_msg):
        self.counter_ += 1
        self.get_logger().info(f"Received synchronized Data. Counter: {self.counter_}")

        if not self.Analyzing:
            self.Analyzing = True
            try:
                # I must merge image info and entity info from topics to prompt it to the VLM

                # Convert the compressed image data from image_msg.data (uint8 array) to bytes
                image_bytes = bytes(image_msg.data)
                # Open the image using PIL and BytesIO to get the dimensions
                with Image.open(BytesIO(image_bytes)) as img:
                    img.load()
                    pixels_width, pixels_height = img.size

                # Convert it to base64 for groq
                if self.use_groq:
                    # Convert the image bytes to a base64 string for Groq
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                self.get_logger().info(f"Image size: {pixels_width}x{pixels_height}")

                # create a string representation of the entity information from entity_msg
                entities_info = "List of entities detcted in this frame (make reference to these exact bounding boxes):\n"

                if not entity_msg.entity_array:
                    entities_info += "No entities detected in this frame.\n"
                else:
                    for entity in entity_msg.entity_array:
                        bbox = entity.bbox_xyxy

                        # Denormalize bboxes 
                        x_min = int(bbox.xmin * pixels_width)
                        y_min = int(bbox.ymin * pixels_height)
                        x_max = int(bbox.xmax * pixels_width)
                        y_max = int(bbox.ymax * pixels_height)

                        # Build phrase with id, label and bounding box (absolute pixel coords).
                        entities_info += f"- ID: {entity.track_id}, Label: {entity.label}, inside bbox: {x_min}, {y_min}, {x_max}, {y_max}\n"
                
                # create a string representation of the human bodies information from human_msg
                human_info = "List of human bodies detected in this frame (make reference to these exact bounding boxes):\n"

                if not human_msg.entity_array:
                    human_info += "No human bodies detected in this frame.\n"
                else:
                    for human in human_msg.entity_array:
                        bbox = human.bbox_xyxy

                        # Denormalize bboxes 
                        x_min = int(bbox.xmin * pixels_width)
                        y_min = int(bbox.ymin * pixels_height)
                        x_max = int(bbox.xmax * pixels_width)
                        y_max = int(bbox.ymax * pixels_height)

                        # Build phrase with id, label and bounding box (absolute pixel coords).
                        human_info += f"- ID: {human.track_id}, Label: {human.label}, inside bbox: {x_min}, {y_min}, {x_max}, {y_max}\n"

                # create a string representation of the skeleton keypoints from skeleton_msg
                skeleton_info = "Human body skeleton keypoints detected in this frame (normalized coordinates 0-1, confidence c):\n"

                if not skeleton_msg.skeleton2d_array:
                    skeleton_info += "No skeleton keypoints detected in this frame.\n"
                else:
                    for skeleton in skeleton_msg.skeleton2d_array:
                        skeleton_info += f"\n  Skeleton ID: {skeleton.skeleton_id} (image: {skeleton.width}x{skeleton.height})\n"
                        for kp in skeleton.skeleton:
                            kp_name = KEYPOINT_NAMES.get(kp.type, f"UNKNOWN_{kp.type}")
                            # Only include keypoints with non-zero confidence
                            if kp.c > 0.0:
                                skeleton_info += f"    - {kp_name}: x={kp.x:.3f}, y={kp.y:.3f}, confidence={kp.c:.2f}\n"

                # ----------------------------------------------------------------
                # 4. Build the Scene Graph prompt for Qwen3.6-27b  (tool-calling mode)
                # ----------------------------------------------------------------
                # With tool calling active the schema is already encoded in the tool
                # definition, so the prompt no longer needs to repeat it. Instead:
                #   a) system_prompt — short role + visual-reasoning mandate
                #   b) user prompt   — sensor data dump + explicit fusion guidance
                #                      (how to combine image evidence with tracker data)
                # The model populates the tool arguments; any CoT stays in content
                # and is automatically ignored by _call_groq.
                # ----------------------------------------------------------------

                system_msg = (
                    "You are a visual perception module on a mobile robot. "
                    "You receive a camera frame together with structured sensor data "
                    "(bounding boxes from an object tracker and skeleton keypoints "
                    "from a pose estimator). "
                    "Your job is to analyse the image carefully and call the "
                    "create_scene_graph tool with a complete, accurate scene graph. "
                    "Prioritise what you can directly see in the image; use the sensor "
                    "data to confirm entity identities, precise locations, and human pose."
                )

                bb_prompt = f"""\
                Analyse the camera frame and the sensor data below, then call `create_scene_graph`.

                --- SENSOR DATA (IMAGE: {pixels_width}x{pixels_height} px) ---
                {entities_info}
                {human_info}
                {skeleton_info}
                --------------------------------------------------------------

                GUIDELINES FOR FILLING THE TOOL ARGUMENTS:

                Entities
                • Include every entity reported by the tracker. Also add background structural
                    elements clearly visible in the image (walls, tables, counters, doors, etc.)
                    even if not tracked — assign them a new sequential ID starting at 1000.
                • Refine the tracker label using what you see (e.g. "object" → "coffee_mug").
                • box_2d order: [ymin, xmin, ymax, xmax] in absolute pixel integers.
                    Convert the tracker bbox from (xmin, ymin, xmax, ymax) to this order.
                • states: pick the most accurate descriptors from the tool schema enum.
                    Infer object states from visual appearance (open/closed, dirty/clean, etc.).
                • action_description: for humans only — describe the ongoing action in ≤8 words
                    (e.g. "picking up a mug from the table"). Use null for objects/structural.

                Skeleton → Pose inference
                • Nose above hips → standing. Hips and knees at similar height → sitting.
                • Wrist(s) raised above shoulder → reaching / gesturing.
                • Wrist near another entity bbox → interacting / holding.
                • Low confidence keypoints (<0.3) should be ignored.

                Relationships
                • Model every meaningful spatial pair (human–object, object–surface, etc.).
                • Use bbox overlap/containment and visual depth cues to choose the predicate.
                • Human attention direction (nose/neck vector) can give a 'looking_at' edge.
                • Prefer specific predicates (on_top_of, holding) over generic ones (near).

                Call create_scene_graph now with the complete graph for this frame."""

                self.get_logger().info("Sending scene graph request to VLM...")

                # 5. Call the VLM — pass system_prompt + assistant_prefix for Groq,
                #    forced_json_schema for maximum schema conformance,
                #    or extra_body for Nebius thinking control.
                if self.use_nebius:
                    response = self.vlm(
                        text_prompt=bb_prompt,
                        image=image_bytes,
                        force_json=True,
                        system_prompt=system_msg,
                        extra_body={"enable_thinking": False}
                    )
                elif self.use_groq:
                    response = self.vlm(
                        text_prompt=bb_prompt,
                        image=image_base64,
                        system_prompt=system_msg,
                        # qwen3.6-27b on Groq: tool calling not supported with image input,
                        # json_schema response_format also not supported.
                        # Strategy: assistant_prefix forces the reply to open with '{'
                        # so the model cannot prepend free text or <think> blocks.
                        # Any residual <think> blocks are stripped by _strip_think_blocks()
                        # inside vlm_client before _extract_json is called.
                        assistant_prefix="{",
                    )


                self.get_logger().info("VLM Scene Graph response received.")
                print("VLM Scene Graph:\n", response)

                # 6. Parse response — tool calling returns the function arguments
                #    directly as a JSON string; _extract_json handles any edge cases.
                if response is None:
                    raise ValueError("VLM returned None response")
                if isinstance(response, str):
                    clean_response = self.vlm._extract_json(response)
                    self.get_logger().info(f"Extracted JSON (first 120 chars): {clean_response[:120]}")
                    response_data = json.loads(clean_response)
                else:
                    response_data = response
                if not isinstance(response_data, dict):
                    raise ValueError(f"Expected dict JSON response, got {type(response_data)}")

                # 7. Normalise tool output → pipeline schema.
                #    The simplified tool schema uses flat fields (box_2d at entity top-level,
                #    entity_type instead of type) to avoid Groq JSON Schema limitations.
                #    Convert back to the nested format expected by the rest of the pipeline.
                for entity in response_data.get("entities", []):
                    # box_2d: flat → nested in spatial_info
                    if "box_2d" in entity and "spatial_info" not in entity:
                        entity["spatial_info"] = {"box_2d": entity.pop("box_2d")}
                    # entity_type → type
                    if "entity_type" in entity and "type" not in entity:
                        entity["type"] = entity.pop("entity_type")
                    # action_description: "none"/""  → None
                    ad = entity.get("action_description", "")
                    if ad in ("", "none", "None", "null", "N/A"):
                        entity["action_description"] = None


                
                # Save the Scene Graph JSON metadata                                                                                                                          
                json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JSON_FOLDER")                                                                          
                os.makedirs(json_dir, exist_ok=True)                                                                                                                             
                json_path = os.path.join(json_dir, f"scene_graph_{self.counter_}.json")                                                                                          
                                                                                                                                                                                    
                metadata = {                                                                                                                                                     
                    "frame_id": self.counter_,
                    "timestamp_sec": image_msg.header.stamp.sec,
                    "timestamp_nanosec": image_msg.header.stamp.nanosec,
                    "width": pixels_width,
                    "height": pixels_height,
                    "scene_graph": response_data,
                }

                with open(json_path, "w") as json_file:
                    json.dump(metadata, json_file, indent=4)

                self.get_logger().info(f"Saved Scene Graph to {json_path}")
                self.vlm.log_metrics()

                # =======================================================
                # 10. Publish the Scene Graph to the LLM Decision Maker
                # =======================================================
                msg = String()
                # Publish the full metadata dict (includes frame details) as a JSON string
                msg.data = json.dumps(metadata) 

                self.scene_graph_pub.publish(msg)
                self.get_logger().info("Published Scene Graph JSON to '/scene_graph'")

            except Exception as e:
                self.get_logger().error(f"Exception in VLM synchronized callback: {format_exc()}")
            finally:
                self.Analyzing = False
        else:
            self.get_logger().info("VLM is busy, skipping frame.")


def main(args=None):
    rclpy.init(args=args)
    node = MultiTopicListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()
