import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import os
import json
from io import BytesIO
from traceback import format_exc
from PIL import Image
import base64
from typing import List, Optional
from pydantic import BaseModel

from std_msgs.msg import String

# Import hri_msgs types
from hri_msgs.msg import EntityArray, Gaze

from src.vlm_client import VLMClient

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroqAPI.env"),
    override=True
)

# ---------------------------------------------------------------------------
# Pydantic schema — documents the expected JSON structure.
# Structured output is enforced via prompt + assistant prefix ("{")
# which forces Qwen to open a JSON object (same strategy as original code).
# ---------------------------------------------------------------------------

class FramePosition(BaseModel):
    region: str          # e.g. "center", "top-left", "bottom-right", ...
    bbox_px: List[int]   # [xmin, ymin, xmax, ymax] absolute pixels

class HRIActivitySchema(BaseModel):
    frame_id: int
    timestamp_sec: int
    timestamp_nanosec: int
    image_width: int
    image_height: int
    person_detected: bool
    frame_position: Optional[FramePosition]
    body_posture: str          # standing | sitting | crouching | walking | unknown
    hand_gesture: str          # ok | stop | thumbs_up | pointing | waving | none | ...
    gaze_direction: str        # forward | left | right | up | down | camera | unknown
    gaze_target: str           # sender/receiver from Gaze msg
    activity_description: str  # free-form summary sentence


class HRITopicListener(Node):
    """
    ROS2 node that subscribes to exactly three topics:
      - /camera/image_raw/compressed  (sensor_msgs/CompressedImage)
      - /humans/detected              (hri_msgs/EntityArray)
      - /humans/faces/gaze            (hri_msgs/Gaze)

    The three topics are time-synchronized via ApproximateTimeSynchronizer.
    On every synchronized set, the merged data is sent to Qwen3.6-27b (Groq)
    which outputs a compact JSON describing the person's frame position,
    body posture, hand gesture, and gaze direction.

    Output JSON files are saved to:
      <script_dir>/HRI_Gesture_and_pose/hri_activity_<frame_id>.json
    """

    def __init__(self):
        super().__init__("hri_topic_listener")
        self.counter_ = 0
        self.Analyzing = False

        # QoS profile compatible with rosbag-recorded BEST_EFFORT sensor topics.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # --- Subscribers ----------------------------------------------------
        self.image_sub = Subscriber(
            self,
            CompressedImage,
            "/camera/image_raw/compressed",
            qos_profile=sensor_qos,
        )
        self.humans_sub = Subscriber(
            self,
            EntityArray,
            "/humans/detected",
            qos_profile=sensor_qos,
        )
        self.gaze_sub = Subscriber(
            self,
            Gaze,
            "/humans/faces/gaze",
            qos_profile=sensor_qos,
        )
        # --------------------------------------------------------------------

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.humans_sub, self.gaze_sub],
            queue_size=10,
            slop=0.5,
        )
        self.sync.registerCallback(self.synchronized_callback)
        self.get_logger().info(
            "Subscribed and synchronized: "
            "/camera/image_raw/compressed, /humans/detected, /humans/faces/gaze."
        )

        # Publisher — keeps compatibility with downstream consumers
        self.activity_pub = self.create_publisher(String, "/hri_activity", 10)

        # Configure and initialize VLM Client (same Qwen model as original code)
        self.model_parameters = self._build_model_parameters()
        self.get_logger().info(
            f"Initializing VLMClient with model: {self.model_parameters['model_name']}"
        )
        self.vlm = VLMClient(**self.model_parameters)
        self.get_logger().info(
            "VLMClient initialized. Node is spinning and waiting for synchronized messages..."
        )

    # ------------------------------------------------------------------
    # Model configuration  (same Qwen model as original code)
    # ------------------------------------------------------------------
    def _build_model_parameters(self) -> dict:
        return {
            "model_name": "groq/qwen3.6-27b",
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1.0,
            "reasoning_effort": "none",   # disables <think> blocks → clean JSON
        }

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _denorm_bbox(bbox, img_w: int, img_h: int) -> tuple:
        """
        Convert NormalizedRegionOfInterest2D (xmin/ymin/xmax/ymax in [0,1])
        to absolute pixel coordinates (xmin, ymin, xmax, ymax).
        """
        x_min = int(bbox.xmin * img_w)
        y_min = int(bbox.ymin * img_h)
        x_max = int(bbox.xmax * img_w)
        y_max = int(bbox.ymax * img_h)
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _classify_frame_region(x_min: int, y_min: int, x_max: int, y_max: int,
                                img_w: int, img_h: int) -> str:
        """
        Classify where the person's bounding-box centroid sits inside a 3x3
        grid of the image (top-left, top-center, top-right, center-left,
        center, center-right, bottom-left, bottom-center, bottom-right).
        """
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        col = ("left"   if cx < img_w / 3.0
               else ("right" if cx > 2.0 * img_w / 3.0 else "center"))
        row = ("top"    if cy < img_h / 3.0
               else ("bottom" if cy > 2.0 * img_h / 3.0 else "center"))
        if row == "center" and col == "center":
            return "center"
        if row == "center":
            return col
        return f"{row}-{col}"

    # ------------------------------------------------------------------
    # Build the human-detection info block for the prompt
    # ------------------------------------------------------------------
    def _build_humans_info(self, humans_msg: EntityArray,
                           img_w: int, img_h: int) -> str:
        """
        Build a compact text block describing every detected person entity.
        Bounding boxes are denormalised to absolute pixel coordinates.
        """
        lines = ["[humans/detected]"]
        if not humans_msg.entity_array:
            lines.append("  (no humans detected in this frame)")
        else:
            for entity in humans_msg.entity_array:
                bbox = entity.bbox_xyxy
                x_min, y_min, x_max, y_max = self._denorm_bbox(bbox, img_w, img_h)
                region = self._classify_frame_region(
                    x_min, y_min, x_max, y_max, img_w, img_h
                )
                conf = getattr(bbox, "c", None)
                conf_str = f"  confidence={conf:.3f}" if conf is not None else ""
                lines.append(
                    f"  - track_id={entity.track_id}"
                    f"  label={entity.label!r}"
                    f"  bbox_px=[{x_min},{y_min},{x_max},{y_max}]"
                    f"  frame_region={region!r}"
                    f"{conf_str}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build the gaze info block for the prompt
    # ------------------------------------------------------------------
    @staticmethod
    def _build_gaze_info(gaze_msg: Gaze) -> str:
        """
        Extract sender (the person/face ID) and receiver (gaze target)
        from hri_msgs/Gaze. Both fields are plain strings.
        An empty receiver means the gaze target is unknown/undetermined.
        """
        sender   = gaze_msg.sender   if gaze_msg.sender   else "unknown"
        receiver = gaze_msg.receiver if gaze_msg.receiver else "unknown"
        lines = [
            "[humans/faces/gaze]",
            f"  sender (person/face ID) : {sender}",
            f"  receiver (gaze target)  : {receiver}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Synchronized callback
    # ------------------------------------------------------------------
    def synchronized_callback(
        self,
        image_msg: CompressedImage,
        humans_msg: EntityArray,
        gaze_msg: Gaze,
    ):
        self.counter_ += 1
        self.get_logger().info(
            f"Received synchronized data — frame #{self.counter_}"
        )

        if not self.Analyzing:
            self.Analyzing = True
            try:
                # ------------------------------------------------------------
                # 1. Decode compressed image  →  dimensions + base64
                # ------------------------------------------------------------
                image_bytes = bytes(image_msg.data)
                with Image.open(BytesIO(image_bytes)) as img:
                    img.load()
                    pixels_width, pixels_height = img.size

                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                self.get_logger().info(
                    f"Image decoded: {pixels_width}x{pixels_height} px"
                )

                # ------------------------------------------------------------
                # 2. Build sensor data text blocks
                # ------------------------------------------------------------
                humans_info = self._build_humans_info(
                    humans_msg, pixels_width, pixels_height
                )
                gaze_info = self._build_gaze_info(gaze_msg)

                self.get_logger().info(
                    f"Humans detected: {len(humans_msg.entity_array)} | "
                    f"Gaze sender={gaze_msg.sender!r}  receiver={gaze_msg.receiver!r}"
                )

                # ------------------------------------------------------------
                # 3. Build the HRI activity analysis prompt for Qwen3.6-27b
                # ------------------------------------------------------------
                system_msg = (
                    "You are a visual HRI (Human-Robot Interaction) perception module. "
                    "You receive a camera frame of a single person in an empty room, "
                    "together with structured sensor data from a human detection pipeline "
                    "and a gaze estimation module. "
                    "Your task is to analyse the image and the sensor data and produce "
                    "a compact, accurate JSON description of the person's current state: "
                    "where they are in the frame, their body posture, what hand gesture "
                    "they are performing, and in which direction they are looking. "
                    "Do NOT build a scene graph. Do NOT add relationships to objects. "
                    "Focus exclusively on the single person visible in the scene. "
                    "Output ONLY valid JSON — no markdown fences, no extra text."
                )

                bb_prompt = (
                    f"Analyse the camera frame and the structured sensor data below, "
                    f"then produce the JSON activity record.\n\n"
                    f"--- SENSOR DATA (image size: {pixels_width}x{pixels_height} px) ---\n\n"
                    f"{humans_info}\n\n"
                    f"{gaze_info}\n\n"
                    f"--- END SENSOR DATA ---\n\n"
                    f"Use the image as the primary source of evidence.  "
                    f"The sensor data provides bounding-box and gaze-direction anchors "
                    f"that you must cross-check against what you actually see.\n\n"
                    f"========================================================================\n"
                    f"DEFINITIONS & ALLOWED VALUES\n"
                    f"========================================================================\n\n"
                    f"body_posture   : Pick exactly ONE from →\n"
                    f"  standing | sitting | crouching | kneeling | walking | lying_down | unknown\n\n"
                    f"hand_gesture   : Pick exactly ONE from →\n"
                    f"  ok | stop | thumbs_up | thumbs_down | pointing | waving | peace | fist |\n"
                    f"  open_hand | crossed_arms | hands_on_hips | clapping | none | unknown\n\n"
                    f"gaze_direction : Pick exactly ONE from →\n"
                    f"  forward | left | right | up | down | camera | unknown\n\n"
                    f"frame_position.region : The 3x3 grid cell where the person's centroid falls →\n"
                    f"  top-left | top-center | top-right |\n"
                    f"  center-left | center | center-right |\n"
                    f"  bottom-left | bottom-center | bottom-right\n\n"
                    f"========================================================================\n"
                    f"INSTRUCTIONS\n"
                    f"========================================================================\n"
                    f"1. Person location  : Use the bbox from /humans/detected (converted to absolute px)\n"
                    f"   to determine which region of the {pixels_width}x{pixels_height} frame the person\n"
                    f"   occupies. Report it in frame_position.region and frame_position.bbox_px.\n"
                    f"2. Body posture     : Inspect the full body visible in the frame and pick one posture label.\n"
                    f"3. Hand gesture     : Look carefully at both hands. If a gesture is visible, pick the best\n"
                    f"   matching label. Use 'none' if no intentional gesture is being performed.\n"
                    f"4. Gaze direction   : Combine the gaze receiver from /humans/faces/gaze with the\n"
                    f"   visual evidence (head orientation, eye direction) to infer the gaze_direction label.\n"
                    f"   Set gaze_target to the receiver string from the sensor data.\n"
                    f"5. activity_description : Write a single concise English sentence (max 20 words)\n"
                    f"   that summarises what the person is doing right now.\n"
                    f"6. If no person is detected, set person_detected=false and fill remaining fields\n"
                    f"   with 'unknown' / null.\n\n"
                    f"========================================================================\n"
                    f"OUTPUT JSON FORMAT  (output ONLY this JSON, nothing else)\n"
                    f"========================================================================\n"
                    f"{{\n"
                    f"  \"frame_id\": {self.counter_},\n"
                    f"  \"timestamp_sec\": {image_msg.header.stamp.sec},\n"
                    f"  \"timestamp_nanosec\": {image_msg.header.stamp.nanosec},\n"
                    f"  \"image_width\": {pixels_width},\n"
                    f"  \"image_height\": {pixels_height},\n"
                    f"  \"person_detected\": <bool>,\n"
                    f"  \"frame_position\": {{\n"
                    f"    \"region\": \"<string: 3x3 grid cell>\",\n"
                    f"    \"bbox_px\": [<int: xmin>, <int: ymin>, <int: xmax>, <int: ymax>]\n"
                    f"  }},\n"
                    f"  \"body_posture\": \"<string: one of the allowed values>\",\n"
                    f"  \"hand_gesture\": \"<string: one of the allowed values>\",\n"
                    f"  \"gaze_direction\": \"<string: one of the allowed values>\",\n"
                    f"  \"gaze_target\": \"<string: receiver from gaze sensor, or 'unknown'>\",\n"
                    f"  \"activity_description\": \"<string: one concise sentence>\"\n"
                    f"}}\n"
                )

                # ------------------------------------------------------------
                # 4. Call Qwen3.6-27b via Groq
                #    assistant_prefix="{" forces the model to start with a JSON
                #    object — same technique as the original code.
                # ------------------------------------------------------------
                response = self.vlm(
                    text_prompt=bb_prompt,
                    image=image_base64,
                    system_prompt=system_msg,
                    assistant_prefix="{",
                    reasoning_effort=self.model_parameters.get("reasoning_effort", "none"),
                )

                self.get_logger().info("VLM HRI-activity response received.")
                self.get_logger().info(
                    f"RAW VLM response (first 300 chars): {str(response)[:300]!r}"
                )
                print("VLM HRI Activity (raw):\n", response)

                # ------------------------------------------------------------
                # 5. Parse the response
                # ------------------------------------------------------------
                if response is None:
                    raise ValueError("VLM returned None response")
                if isinstance(response, str):
                    clean_response = self.vlm._extract_json(response)
                    self.get_logger().info(
                        f"Extracted JSON (first 120 chars): {clean_response[:120]}"
                    )
                    response_data = json.loads(clean_response)
                else:
                    response_data = response

                if not isinstance(response_data, dict):
                    raise ValueError(
                        f"Expected dict JSON response, got {type(response_data)}"
                    )

                # ------------------------------------------------------------
                # 6. Normalise optional / aliased fields
                # ------------------------------------------------------------
                # If no person detected, frame_position may be absent
                if not response_data.get("person_detected", True):
                    response_data.setdefault("frame_position", None)

                # Coerce string "null"/"none" in activity_description → empty string
                ad = response_data.get("activity_description", "")
                if ad in ("null", "none", "None", "N/A", ""):
                    response_data["activity_description"] = ""

                # ------------------------------------------------------------
                # 7. Save JSON to HRI_Gesture_and_pose/
                # ------------------------------------------------------------
                json_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "OutputData", "HRI_Gesture_and_pose",
                )
                os.makedirs(json_dir, exist_ok=True)
                json_path = os.path.join(
                    json_dir, f"hri_activity_{self.counter_}.json"
                )

                with open(json_path, "w") as json_file:
                    json.dump(response_data, json_file, indent=4)

                self.get_logger().info(f"Saved HRI activity JSON to {json_path}")
                self.vlm.log_metrics()

                # ------------------------------------------------------------
                # 8. Publish to /hri_activity topic
                # ------------------------------------------------------------
                msg = String()
                msg.data = json.dumps(response_data)
                self.activity_pub.publish(msg)
                self.get_logger().info("Published HRI activity JSON to '/hri_activity'")

            except Exception:
                self.get_logger().error(
                    f"Exception in VLM synchronized callback:\n{format_exc()}"
                )
            finally:
                self.Analyzing = False
        else:
            self.get_logger().info("VLM is busy, skipping frame.")


def main(args=None):
    rclpy.init(args=args)
    node = HRITopicListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
