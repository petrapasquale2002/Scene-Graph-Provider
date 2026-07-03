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

# Import EntityArray for entity tracking
from hri_msgs.msg import EntityArray

from src.vlm_client import VLMClient

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroqAPI.env"),
    override=True
)

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

# ---------------------------------------------------------------------------

class MultiTopicListener(Node):
    # This listener subscribes to the compressed image, entity detection, and human detection topics.
    # It synchronizes them and sends merged data to the VLM to generate a detailed Scene Graph.
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
        # self.human_sub = Subscriber(
        #     self,
        #     EntityArray,
        #     "/humans/detected"
        # )
   

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.entity_sub],# self.human_sub],
            queue_size=10,
            slop=0.5
        )
        self.sync.registerCallback(self.synchronized_callback)
        self.get_logger().info("Subscribed and synchronized image and entities.")

        # Create a publisher to send the Scene Graph to the LLM Decision Maker                                                                                                   
        self.scene_graph_pub = self.create_publisher(                                                                                                                            
            String,                                                                                                                                                              
            '/scene_graph',                                                                                                                                                      
            10                                                                                                                                                                   
        )     

        # Configure and Initialize VLM Client
        self.model_parameters = self.test_groq_vlm()
        self.get_logger().info(f"Initializing VLMClient with model: {self.model_parameters['model_name']}")
        self.vlm = VLMClient(**self.model_parameters)
        self.get_logger().info("VLMClient initialized successfully. Node is now spinning and waiting for synchronized messages...")

    def test_groq_vlm(self):
        return {
            "model_name": "groq/qwen3.6-27b",
            'temperature': 0.0,
            'max_tokens': 4096,  # Increased from 1500 to 4096 to prevent truncation
            'top_p': 1.0,
            'reasoning_effort': 'none',  # Disabilita il thinking mode di Qwen3 per avere output diretto
        }

    def synchronized_callback(self, image_msg, entity_msg):#, human_msg):
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

                # Convert the image bytes to a base64 string
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
                
                # # create a string representation of the human bodies information from human_msg
                # human_info = "List of human bodies detected in this frame (make reference to these exact bounding boxes):\n"

                # if not human_msg.entity_array:
                #     human_info += "No human bodies detected in this frame.\n"
                # else:
                #     for human in human_msg.entity_array:
                #         bbox = human.bbox_xyxy

                #         # Denormalize bboxes 
                #         x_min = int(bbox.xmin * pixels_width)
                #         y_min = int(bbox.ymin * pixels_height)
                #         x_max = int(bbox.xmax * pixels_width)
                #         y_max = int(bbox.ymax * pixels_height)

                #         # Build phrase with id, label and bounding box (absolute pixel coords).
                #         human_info += f"- ID: {human.track_id}, Label: {human.label}, inside bbox: {x_min}, {y_min}, {x_max}, {y_max}\n"

                # ----------------------------------------------------------------
                # 4. Build the Scene Graph prompt for Qwen3.6-27b 
                # ----------------------------------------------------------------
                # Povide the VLM a prompt that gives the json schema
                # definition to follow strictly. 
                # ----------------------------------------------------------------

                system_msg = (
                    "You are a visual perception module on a mobile robot. "
                    "You receive a camera frame together with structured sensor data "
                    "Your job is to analyse the image carefully and call the "
                    "create_scene_graph tool with a complete, accurate scene graph. "
                    "Prioritise what you get from the message data and confirm the relationships with the image; "
                    "Follow a json schema as output."
                )

                bb_prompt = f"""\
                Analyse the camera frame and the sensor data below, then call `create_scene_graph`.

                --- SENSOR DATA (IMAGE: {pixels_width}x{pixels_height} px) ---
                {entities_info}
                (MISSING HUMAN INFO BUT IGNORE FOR NOW)
                --------------------------------------------------------------

                Use the raw image just as a reference for scene graph generation output.
                Your goal is to generate a comprehensive, physically-grounded Scene Graph.
                The output must serve as a deterministic spatial and semantic map for a downstream LLM decision-making agent designed for social robotics and human-robot interaction.

                ------------------------------------------------------------------------
                ALLOWED STATES
                ------------------------------------------------------------------------
                [Object/Inanimate States]: open, closed, empty, full, dirty, clean, hot, cold, turned_on, turned_off, stable, unstable, broken
                [Human/Agent States]: standing, sitting, walking, reaching, looking_at, interacting, neutral, gesturing (use the image to state the posture and confirm the human's actions)
                [Shared States]: reachable, occluded, held_by, static, moving, unknown

                ------------------------------------------------------------------------
                ALLOWED RELATIONSHIPS (Strictly Directed: Subject -> Predicate -> Object)
                ------------------------------------------------------------------------
                [Topological / Contact]: on_top_of, inside, part_of, touching, not_touching, embedded_in
                [Relative Spatial / Proximity]: next_to, near, above, below, in_front_of, behind, on_the_left_of, on_the_right_of, facing, occluding
                [Agent / Interaction]: holding, held_by, pointed_by, looking_at, operating

                ========================================================================
                LAYOUT CONFIGURATION Context: DOMESTIC LIVING SPACE (Living Room & Dining Area)
                ========================================================================
                Description:
                A cozy domestic environment designed for daily living and social interaction. It features a dining table used for meals, a comfortable sofa for reading and relaxing, and everyday household items scattered around, including dishes, utensils, food, and books. A human user is present, interacting naturally with the environment and the objects.

                Typical Entities & Scene Commonsense:
                - dining_table (type: structural, states: clean, static)
                - sofa (type: structural, states: clean, static)
                - plate (type: object, states: clean, empty, reachable, static | relationship: on_top_of -> dining_table)
                - fork (type: object, states: clean, reachable, static | relationship: next_to -> plate)
                - apple (type: object, states: clean, reachable, static | relationship: inside -> plate)
                - book (type: object, states: closed, static, reachable | relationship: on_top_of -> sofa)
                - human_user (type: human, states: sitting, interacting | relationship: near -> dining_table)

                Example Scene Graph JSON:
                {{
                "entities": [
                    {{
                    "id": 0, "label": "dining_table", "type": "structural", "states": ["clean", "static"], 
                    "spatial_info": {{"box_2d": [200, 100, 500, 600]}}, 
                    "action_description": null
                    }},
                    {{
                    "id": 1, "label": "sofa", "type": "structural", "states": ["clean", "static"], 
                    "spatial_info": {{"box_2d": [150, 600, 400, 900]}}, 
                    "action_description": null
                    }},
                    {{
                    "id": 2, "label": "plate", "type": "object", "states": ["clean", "empty", "reachable", "static"], 
                    "spatial_info": {{"box_2d": [210, 250, 260, 350]}}, 
                    "action_description": null
                    }},
                    {{
                    "id": 3, "label": "fork", "type": "object", "states": ["clean", "reachable", "static"], 
                    "spatial_info": {{"box_2d": [215, 360, 225, 420]}}, 
                    "action_description": null
                    }},
                    {{
                    "id": 4, "label": "apple", "type": "object", "states": ["clean", "reachable", "static"], 
                    "spatial_info": {{"box_2d": [220, 280, 250, 320]}}, 
                    "action_description": null
                    }},
                    {{
                    "id": 5, "label": "book", "type": "object", "states": ["closed", "static", "reachable"], 
                    "spatial_info": {{"box_2d": [180, 650, 220, 720]}}, 
                    "action_description": null
                    }},
                    {{
                    "id": 6, "label": "human_user", "type": "human", "states": ["sitting", "interacting"], 
                    "spatial_info": {{"box_2d": [100, 150, 450, 300]}}, 
                    "action_description": "sitting at the table and reaching for the apple"
                    }}
                ],
                "relationships": [
                    {{"subject_id": 2, "predicate": "on_top_of", "object_id": 0}},
                    {{"subject_id": 3, "predicate": "on_top_of", "object_id": 0}},
                    {{"subject_id": 3, "predicate": "next_to", "object_id": 2}},
                    {{"subject_id": 4, "predicate": "inside", "object_id": 2}},
                    {{"subject_id": 5, "predicate": "on_top_of", "object_id": 1}},
                    {{"subject_id": 6, "predicate": "near", "object_id": 0}},
                    {{"subject_id": 6, "predicate": "looking_at", "object_id": 4}}
                ]
                }}

                ------------------------------------------------------------------------
                INSTRUCTIONS
                ------------------------------------------------------------------------
                1. Entity Identification: Detect all key entities (everyday objects, household architectural elements, humans, specific body parts if heavily interacting).
                2. Physical Commonsense & Grounding: Ground your reasoning in physical reality. Furniture sits on the floor; food goes on plates or tables; humans sit on chairs/sofas or stand on the floor. Do not hallucinate floating or physically impossible states.
                3. State Assignment: Apply states based on the entity type (Inanimate vs Human vs Shared). Pay special attention to human social cues (gesturing, interacting, looking_at).
                4. Spatial & Relative Relationships: Deduce precise relative positions. If Object A is to the left of Object B from the camera perspective, log [A -> on_the_left_of -> B]. If Bounding Box data is deducible, ensure relationships strictly mirror the spatial vectors.
                5. JSON Formatting: The final output must be a single, valid JSON object starting with {{ and ending with }}. Do not include any markdown block formatting (like ```json) around the JSON.
                6. Reasoning: If you must reason or explain, do it in a <think>...</think> block at the very beginning of your response, or do it as plain text before the JSON block. Do not include any text, reasoning, or explanations after the closing brace }} of the JSON block.

                ------------------------------------------------------------------------
                OUTPUT JSON FORMAT
                ------------------------------------------------------------------------
                {{
                "entities": [
                    {{
                    "id": <int: unique ID starting from 0>,
                    "label": "<string: entity_name>",
                    "type": "<string: 'object' | 'human' | 'structural'>",
                    "states": [<string: chosen from allowed states>],
                    "spatial_info": {{
                        "box_2d": [<int: ymin>, <int: xmin>, <int: ymax>, <int: xmax>]
                    }},
                    "action_description": "<string: specific action verb if human (e.g., 'reading a book', 'pointing at the fork'), otherwise null>"
                    }}
                ],
                "relationships": [
                    {{
                    "subject_id": <int: ID of the subject entity>,
                    "predicate": "<string: predicate from allowed relationships>",
                    "object_id": <int: ID of the object entity>
                    }}
                ]
                }}
                """
             
            
                response = self.vlm(
                    text_prompt=bb_prompt,
                    image=image_base64,
                    system_prompt=system_msg,
                    assistant_prefix="{",
                    reasoning_effort=self.model_parameters.get('reasoning_effort', 'none'),
                )


                self.get_logger().info("VLM Scene Graph response received.")
                self.get_logger().info(f"RAW VLM response (first 300 chars): {str(response)[:300]!r}")
                print("VLM Scene Graph (raw):\n", response)

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
                json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OutputData/Scene_Graph_only_entities")                                                                          
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
