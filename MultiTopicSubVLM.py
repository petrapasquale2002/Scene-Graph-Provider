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

from std_msgs.msg import String

# Import EntityArray for entity tracking
from hri_msgs.msg import EntityArray

from src.vlm_client import VLMClient

class MultiTopicListener(Node):
    # This listener subscribes to both the compressed image topic and the human body skeleton topic.
    # It synchronizes the incoming messages, extracts their contents, and sends them to the VLM
    # (Qwen via Nebius) to generate a detailed Scene Graph with relationships and states.
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

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.entity_sub, self.human_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_callback)
        self.get_logger().info("Subscribed and synchronized image & entity topics.")

        # 3. Configure and Initialize VLM Client
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
            "model_name": "groq/llama4-scout-17b",
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.9
        }

    def test_nebius_vlm(self):
        return {
            "model_name": "nebius/qwen3-2.5-70b",
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.9,
        }

    def synchronized_callback(self, image_msg, entity_msg, human_msg):
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
                
                #original size of the image
                #orig_width, orig_height = pixels_width, pixels_height

                # Reduce image size but maintain the aspect ratio, to reduce visual tokens sent
                #scale_factor = 1.0
                    
                # Resize
                #img_resized = img.resize((pixels_width, pixels_height), Image.Resampling.LANCZOS)
                
                # Save the resized image in bytes to be sent to the VLM
                #buffer = BytesIO()
                #img_resized.save(buffer, format="JPEG")
                #image_bytes = buffer.getvalue()

                # Convert it to base64 for groq
                if self.use_groq:
                    # Convert the image bytes to a base64 string for Groq
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                #self.get_logger().info(f"Resized image to {pixels_height} successfully ({pixels_width}x{pixels_height}). Scale factor: {scale_factor:.2f}")

                # create a string representation of the entity information from entity_msg
                entities_info = "List of entities detcted in this frame (make reference to these exact bounding boxes):\n"

                if not entity_msg.entity_array:
                    entities_info += "No entities detected in this frame.\n"
                else:
                    for entity in entity_msg.entity_array:
                        #bbox = entity.bbox_xyxy

                        # Rescale bounding box coordinates according to the new image size.
                        #x_min = int(bbox.xmin * scale_factor)
                        #y_min = int(bbox.ymin * scale_factor)
                        #x_max = int(bbox.xmax * scale_factor)
                        #y_max = int(bbox.ymax * scale_factor)

                        # Build phrase with id, label and bounding box.
                        entities_info += f"- ID: {entity.track_id}, Label: {entity.label}\n"
                
                # create a string representation of the human bodies information from human_msg
                human_info = "List of human bodies detected in this frame (make reference to these exact bounding boxes):\n"

                if not human_msg.entity_array:
                    human_info += "No human bodies detected in this frame.\n"
                else:
                    for human in human_msg.entity_array:
                        #bbox = human.bbox_xyxy

                        # Rescale bounding box coordinates according to the new image size.
                        #x_min = int(bbox.xmin * scale_factor)
                        #y_min = int(bbox.ymin * scale_factor)
                        #x_max = int(bbox.xmax * scale_factor)
                        #y_max = int(bbox.ymax * scale_factor)

                        # Build phrase with id, label and bounding box.
                        human_info += f"- ID: {human.track_id}, Label: {human.label}\n"

                # 4. Prepare the Scene Graph prompt
                task = "Construct a detailed Scene Graph from the image and skeleton data."
                bb_prompt = f"""
                Task: {task}
                Image Dimensions: {pixels_width} x {pixels_height}

                Analyze the raw image using the provided contextual data (Entity Info: {entities_info}, Human Info: {human_info}). Your goal is to generate a comprehensive, physically-grounded Scene Graph. 

                The output must serve as a deterministic spatial and semantic map for a downstream LLM decision-making agent designed for social robotics and human-robot interaction.

                ------------------------------------------------------------------------
                ALLOWED STATES
                ------------------------------------------------------------------------
                [Object/Inanimate States]: open, closed, empty, full, dirty, clean, hot, cold, turned_on, turned_off, stable, unstable, broken
                [Human/Agent States]: standing, sitting, walking, reaching, looking_at, interacting, neutral, gesturing
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
                5. JSON Formatting: Output MUST be a single, valid JSON object. Do not include any markdown block formatting (like ```json), explanations, or trailing text.

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

                self.get_logger().info("Sending scene graph request to VLM...")
                
                # 5. Call the VLM client for Qwen
                if self.use_nebius:
                    response = self.vlm(
                        text_prompt=bb_prompt,
                        image=image_bytes,
                        force_json_response=True
                    )
                elif self.use_groq:
                    response = self.vlm(
                        text_prompt=bb_prompt,
                        image=image_base64,
                        force_json_response=True
                    )

                self.get_logger().info("VLM Scene Graph response received.")
                print("VLM Scene Graph:\n", response)

                # 6. Parse response
                if response is None:
                    raise ValueError("VLM returned None response")
                response_data = json.loads(response) if isinstance(response, str) else response
                if not isinstance(response_data, dict):
                    raise ValueError(f"Expected dict JSON response, got {type(response_data)}")

                # 7. Draw bounding boxes
                # Map the VLM response format's "bounding_box" key to the "coordinates" key expected by _draw_bbs
                #drawing_boxes = []
                #for ent in response_data.get("entities", []):
                #    if "bounding_box" in ent:
                #        drawing_boxes.append({
                #            "label": ent.get("label", ""),
                #            "coordinates": ent["bounding_box"]
                #        })
                #    else:
                #        drawing_boxes.append(ent)

                #with Image.open(BytesIO(image_bytes)) as img:
                #    annotated_img = self.vlm._draw_bbs(drawing_boxes, img, print=False)

                # 8. Save the annotated image
                #image_pool_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scene_Graph_Image_pool")
                #os.makedirs(image_pool_dir, exist_ok=True)
                #output_path = os.path.join(image_pool_dir, f"annotated_frame_{self.counter_}.jpg")
                #annotated_img.save(output_path)

                # 9. Save the Scene Graph JSON metadata
                json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scene_Graph_json")
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
