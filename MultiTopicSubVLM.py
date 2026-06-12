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
                orig_width, orig_height = pixels_width, pixels_height

                # Reduce image size but maintain the aspect ratio, to reduce visual tokens sent
                pixels_height = 480
                scale_factor = pixels_height / orig_height
                pixels_width = int(orig_width * scale_factor)
                    
                # Resize
                img_resized = img.resize((pixels_width, pixels_height), Image.Resampling.LANCZOS)
                
                # Save the resized image in bytes to be sent to the VLM
                buffer = BytesIO()
                img_resized.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()

                # Convert it to base64 for groq
                if self.use_groq:
                    # Convert the image bytes to a base64 string for Groq
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                self.get_logger().info(f"Resized image to {pixels_height} successfully ({pixels_width}x{pixels_height}). Scale factor: {scale_factor:.2f}")

                # create a string representation of the entity information from entity_msg
                entities_info = "List of entities detcted in this frame (make reference to these exact bounding boxes):\n"

                if not entity_msg.entity_array:
                    entities_info += "No entities detected in this frame.\n"
                else:
                    for entity in entity_msg.entity_array:
                        bbox = entity.bbox_xyxy

                        # Rescale bounding box coordinates according to the new image size.
                        x_min = int(bbox.xmin * scale_factor)
                        y_min = int(bbox.ymin * scale_factor)
                        x_max = int(bbox.xmax * scale_factor)
                        y_max = int(bbox.ymax * scale_factor)

                        # Build phrase with id, label and bounding box.
                        entities_info += f"- ID: {entity.track_id}, Label: {entity.label}, Bounding Box: [xmin: {x_min}, ymin: {y_min}, xmax: {x_max}, ymax: {y_max}]\n"
                
                # create a string representation of the human bodies information from human_msg
                human_info = "List of human bodies detected in this frame (make reference to these exact bounding boxes):\n"

                if not human_msg.entity_array:
                    human_info += "No human bodies detected in this frame.\n"
                else:
                    for human in human_msg.entity_array:
                        bbox = human.bbox_xyxy

                        # Rescale bounding box coordinates according to the new image size.
                        x_min = int(bbox.xmin * scale_factor)
                        y_min = int(bbox.ymin * scale_factor)
                        x_max = int(bbox.xmax * scale_factor)
                        y_max = int(bbox.ymax * scale_factor)

                        # Build phrase with id, label and bounding box.
                        human_info += f"- ID: {human.track_id}, Label: {human.label}, Bounding Box: [xmin: {x_min}, ymin: {y_min}, xmax: {x_max}, ymax: {y_max}]\n"

                # 4. Prepare the Scene Graph prompt
                task = "Construct a detailed Scene Graph from the image and skeleton data."
                bb_prompt = f"""
                Task: {task}
                Image Dimensions: {pixels_width} x {pixels_height}


                Use the visual details of the raw image combined with the entity information {entities_info} and human information {human_info} to boundary box all the entities, both inanimated and humans, classify their states, and deduce relationships.

                ---

                Allowed States for Entities (Select all that apply for each entity):
                [open, closed, empty, full, dirty, clean, reachable, occluded, held, static, moving, unknown, hot, cold]

                Allowed Relationship Types (Directed edge: Subject -> Relationship -> Object):
                [on_top_of, inside, next_to, near, held_by, holding, pointed_by, facing, occluding, part_of, same_instance_as]

                ---

                Instructions:
                1. Identify all key entities in the scene (objects, humans, body parts).
                2. Assign relevant states to each entity from the "Allowed States" list above. Remember that the scenes are based onreality, which means that you will not find entities in unusual places, for example: humans or chairs will be not placed the table, but rather placed or standing on the floor. Use this reasoning for classification of states and relationships.
                3. Establish directed relationships between entities using ONLY the relationship types listed in the "Allowed Relationship Types" list above.
                4. Ensure the output strictly follows the JSON format below. Do not include any markdown block formatting (like ```json), explanations, or trailing text.

                Output JSON Format:
                {{
                "entities": [
                    {{
                    "id": <int: unique ID starting from 0>,
                    "label": "<string: name of the entity>",
                    "states": [<string: list of states chosen from allowed states>],
                    "bounding_box": {{
                        "x_min": <int: pixel coordinate>,
                        "y_min": <int: pixel coordinate>,
                        "x_max": <int: pixel coordinate>,
                        "y_max": <int: pixel coordinate>
                    }}
                    "action description": "<string: only if the entity is a human performing a recognizable action, such as standing, talking, pointing, picking up, cutting ect, otherwise omit this field>"
                    }}
                ],
                "relationships": [
                    {{
                    "subject_id": <int: ID of the subject entity>,
                    "predicate": "<string: relationship type chosen from allowed relationship types>",
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
                drawing_boxes = []
                for ent in response_data.get("entities", []):
                    if "bounding_box" in ent:
                        drawing_boxes.append({
                            "label": ent.get("label", ""),
                            "coordinates": ent["bounding_box"]
                        })
                    else:
                        drawing_boxes.append(ent)

                with Image.open(BytesIO(image_bytes)) as img:
                    annotated_img = self.vlm._draw_bbs(drawing_boxes, img, print=False)

                # 8. Save the annotated image
                image_pool_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scene_Graph_Image_pool")
                os.makedirs(image_pool_dir, exist_ok=True)
                output_path = os.path.join(image_pool_dir, f"annotated_frame_{self.counter_}.jpg")
                annotated_img.save(output_path)

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
