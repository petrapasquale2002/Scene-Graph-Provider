import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from collections import deque
import threading
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
from hri_msgs.msg import EntityArray, PersonArray

# Import TIPS message types
from eut_scene_msgs.msg import (
    TipsObjectIdentityArray,
    TipsEmbeddingArray,
    TipsPatchMatchArray,
)

from src.vlm_client import VLMClient

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroqAPI.env"),
    override=True
)

# These are the attributes for each instance in JSOn schema

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

# Maximum number of contour points to include in the prompt (to keep it concise)
_MAX_CONTOUR_PTS = 12


def _contour_to_pixel_pts(contour, img_w: int, img_h: int, max_pts: int = _MAX_CONTOUR_PTS) -> List[tuple]:
    """
    Convert a list of NormalizedPointOfInterest2D to absolute pixel (x, y) tuples.
    Uniformly sub-samples to at most `max_pts` points so the prompt stays compact.
    Returns an empty list when `contour` is empty.
    """
    if not contour:
        return []
    total = len(contour)
    step = max(1, total // max_pts)
    pts = []
    for i in range(0, total, step):
        p = contour[i]
        pts.append((int(p.x * img_w), int(p.y * img_h)))
        if len(pts) >= max_pts:
            break
    return pts


def _bbox_xcywh_to_abs(xcenter: float, ycenter: float,
                        width: float, height: float,
                        img_w: int, img_h: int) -> tuple:
    """
    Convert normalised (xcenter, ycenter, width, height) -> absolute pixel
    (xmin, ymin, xmax, ymax).
    """
    x_min = int((xcenter - width  / 2.0) * img_w)
    y_min = int((ycenter - height / 2.0) * img_h)
    x_max = int((xcenter + width  / 2.0) * img_w)
    y_max = int((ycenter + height / 2.0) * img_h)
    return x_min, y_min, x_max, y_max


class MultiTopicListener(Node):
    # This listener subscribes to the compressed image, entity detection, and
    # TIPS perception topics.  Image compressed  is the trigger topic: its timestamp is the filter for
    # data retrieval from caches, where every last topic message is stored, for synchronization.
    # The datapack is then provided to theVLM to generate a detailed Scene Graph.
    def __init__(self):
        super().__init__("multi_listener")
        self.counter_ = 0
        self.Analyzing = False

        # QoS profile that matches rosbag-recorded sensor topics (BEST_EFFORT).
        # Without this, ROS2 logs "incompatible QoS / RELIABILITY" and drops all messages.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # -- Slop window (seconds) for timestamp matching --------------------
        self._slop = 0.5
        self._cache_lock = threading.Lock() # for message's persistance

        # Caches for optional topics: deque of (timestamp_float, msg)
        self._entity_cache               = deque(maxlen=10)
        self._human_cache                = deque(maxlen=10)
        self._human_detected_cache       = deque(maxlen=10) 
        self._tips_identities_cache      = deque(maxlen=10)
        self._tips_embeddings_cache      = deque(maxlen=10)
        self._tips_patch_matches_cache   = deque(maxlen=10)

        # -- Image: primary trigger (always published) -----------------------
        self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self._on_image,
            qos_profile=sensor_qos,
        )

        # -- Optional topics: fill caches ------------------------------------
        self.create_subscription(
            EntityArray,
            "/entities/detected",
            lambda msg: self._cache_push(self._entity_cache, msg),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            PersonArray,
            "/humans/unified",
            lambda msg: self._cache_push(self._human_cache, msg),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            EntityArray,
            "/humans/detected",
            lambda msg: self._cache_push(self._human_detected_cache, msg),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            TipsObjectIdentityArray,
            "/tips/object_identities",
            lambda msg: self._cache_push(self._tips_identities_cache, msg),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            TipsEmbeddingArray,
            "/tips/embeddings",
            lambda msg: self._cache_push(self._tips_embeddings_cache, msg),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            TipsPatchMatchArray,
            "/tips/patch_matches",
            lambda msg: self._cache_push(self._tips_patch_matches_cache, msg),
            qos_profile=sensor_qos,
        )
        # --------------------------------------------------------------------

        self.get_logger().info(
            "Subscribed: image (trigger) + optional caches for "
            "entities, /humans/unified, /humans/detected, /tips/object_identities, /tips/embeddings, /tips/patch_matches."
        )

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
            'max_tokens': 4096,
            'top_p': 1.0,
            'reasoning_effort': 'none',
        }

    # ------------------------------------------------------------------
    # Helper: build the {tips ...} string section from the three arrays
    # ------------------------------------------------------------------
    def _build_tips_info( # they are like dependences for the AI model
        self,
        tips_identities_msg: TipsObjectIdentityArray,
        tips_embeddings_msg: TipsEmbeddingArray,
        tips_patch_matches_msg: TipsPatchMatchArray,
        img_w: int,
        img_h: int,
    ) -> str:
        """
        Build a compact human-readable summary of the TIPS perception data
        to be injected as a {tips ...} block inside the VLM prompt.

        All bounding boxes (normalised xcenter/ycenter/width/height) are
        converted to absolute pixel (xmin, ymin, xmax, ymax) coordinates.
        When a segmentation contour is available it is sub-sampled and
        reported as a list of (x,y) pixel points for richer spatial grounding.
        """
        lines: List[str] = ["{tips"]

        # -- 1. Object Identities --------------------------------------------
        lines.append("  [object_identities]  # stable physical-object re-ID")
        if not tips_identities_msg.identities:
            lines.append("    (none)")
        else:
            for ident in tips_identities_msg.identities:
                x_min, y_min, x_max, y_max = _bbox_xcywh_to_abs(
                    ident.bbox_xcenter, ident.bbox_ycenter,
                    ident.bbox_width,   ident.bbox_height,
                    img_w, img_h,
                )
                entry = (
                    f"    - track_id={ident.entity.track_id}"
                    f"  object_id={ident.object_id!r}"
                    f"  category={ident.category!r}"
                    f"  confirmed={ident.is_confirmed}"
                    f"  new={ident.is_new_identity}"
                    f"  confidence={ident.assignment_confidence:.3f}"
                    f"  bbox_px=[{x_min},{y_min},{x_max},{y_max}]"
                )
                # Contour (segmentation mask boundary)
                contour_pts = _contour_to_pixel_pts(
                    ident.entity.contour, img_w, img_h
                )
                if contour_pts:
                    entry += f"  contour_px={contour_pts}"
                lines.append(entry)

        # -- 2. Visual Embeddings summary ------------------------------------
        lines.append("  [embeddings]  # TIPS visual embeddings (category + bbox)")
        if not tips_embeddings_msg.embeddings:
            lines.append("    (none)")
        else:
            for emb in tips_embeddings_msg.embeddings:
                x_min, y_min, x_max, y_max = _bbox_xcywh_to_abs(
                    emb.bbox_xcenter, emb.bbox_ycenter,
                    emb.bbox_width,   emb.bbox_height,
                    img_w, img_h,
                )
                entry = (
                    f"    - track_id={emb.entity.track_id}"
                    f"  category={emb.category!r}"
                    f"  model={emb.tips_model!r}_{emb.variant!r}"
                    f"  embedding_dim={emb.embedding_dim}"
                    f"  bbox_px=[{x_min},{y_min},{x_max},{y_max}]"
                )
                # Contour (segmentation mask boundary)
                contour_pts = _contour_to_pixel_pts(
                    emb.entity.contour, img_w, img_h
                )
                if contour_pts:
                    entry += f"  contour_px={contour_pts}"
                lines.append(entry)

        # -- 3. Patch Matches (text-query spatial grounding) -----------------
        lines.append("  [patch_matches]  # best-matching ViT patch per text query")
        if not tips_patch_matches_msg.matches:
            lines.append("    (none)")
        else:
            for match in tips_patch_matches_msg.matches:
                x_min, y_min, x_max, y_max = _bbox_xcywh_to_abs(
                    match.bbox_xcenter, match.bbox_ycenter,
                    match.bbox_width,   match.bbox_height,
                    img_w, img_h,
                )
                entry = (
                    f"    - track_id={match.entity.track_id}"
                    f"  query={match.query!r}"
                    f"  patch=({match.patch_row},{match.patch_col})"
                    f"  cosine_sim={match.cosine_similarity:.3f}"
                    f"  bbox_px=[{x_min},{y_min},{x_max},{y_max}]"
                )
                # Contour (segmentation mask boundary)
                contour_pts = _contour_to_pixel_pts(
                    match.entity.contour, img_w, img_h
                )
                if contour_pts:
                    entry += f"  contour_px={contour_pts}"
                lines.append(entry)

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _ts(self, msg) -> float:
        """Return the header timestamp of a message as a float (seconds)."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _cache_push(self, cache: deque, msg) -> None:
        """
        Push (reception_time, msg) into a thread-safe deque cache.
        Uses time.monotonic() — the real wall-clock time at which the message
        arrived at this node — instead of the message header timestamp.
        This makes matching robust regardless of rosbag speed or VLM inference delay.
        """
        t_received = time.monotonic()
        with self._cache_lock:
            cache.append((t_received, msg))


    def _closest(self, cache: deque, t_ref: float):
        """
        Return the message in *cache* whose timestamp is closest to *t_ref*
        and within self._slop seconds.  Returns None if the cache is empty
        or no entry falls within the slop window.
        """
        with self._cache_lock:
            if not cache:
                return None
            best, best_dt = None, float("inf")
            for t, msg in cache:
                dt = abs(t - t_ref)
                if dt <= self._slop and dt < best_dt:
                    best, best_dt = msg, dt
            if best is None:
                # Log the closest miss to help tune slop
                closest_dt = min(abs(t - t_ref) for t, _ in cache)
                self.get_logger().debug(
                    f"_closest miss for {type(cache[0][1]).__name__}: "
                    f"best Δt={closest_dt:.3f}s exceeds slop={self._slop}s"
                )
            return best


    def _on_image(self, image_msg) -> None:
        """
        Called on every incoming camera frame.
        Looks up the best-matching message for each optional topic within the
        slop window and forwards them (or empty defaults) to synchronized_callback.
        Uses time.monotonic() as reference — same clock used by _cache_push.
        """
        t_now = time.monotonic()

        entity_msg             = self._closest(self._entity_cache,             t_now) or EntityArray()
        human_msg              = self._closest(self._human_cache,              t_now) or PersonArray()
        human_detected_msg     = self._closest(self._human_detected_cache,     t_now) or EntityArray()
        tips_identities_msg    = self._closest(self._tips_identities_cache,    t_now) or TipsObjectIdentityArray()
        tips_embeddings_msg    = self._closest(self._tips_embeddings_cache,    t_now) or TipsEmbeddingArray()
        tips_patch_matches_msg = self._closest(self._tips_patch_matches_cache, t_now) or TipsPatchMatchArray()

        # Diagnostic: log cache sizes and match results
        with self._cache_lock:
            self.get_logger().info(
                f"Cache state (mono={t_now:.3f}s) | "
                f"entities={len(self._entity_cache)}(match={'YES' if entity_msg.entity_array else 'empty/no'}), "
                f"humans={len(self._human_cache)}(match={'YES' if human_msg.persons else 'empty/no'}), "
                f"human_detected={len(self._human_detected_cache)}(match={'YES' if human_detected_msg.entity_array else 'empty/no'}), "
                f"tips_id={len(self._tips_identities_cache)}(match={'YES' if tips_identities_msg.identities else 'empty/no'}), "
                f"tips_emb={len(self._tips_embeddings_cache)}(match={'YES' if tips_embeddings_msg.embeddings else 'empty/no'}), "
                f"tips_pm={len(self._tips_patch_matches_cache)}(match={'YES' if tips_patch_matches_msg.matches else 'empty/no'})"
            )

        self.synchronized_callback(
            image_msg,
            entity_msg,
            human_msg,
            human_detected_msg,
            tips_identities_msg,
            tips_embeddings_msg,
            tips_patch_matches_msg,
        )


    # ------------------------------------------------------------------
    # Synchronized callback
    # ------------------------------------------------------------------
    def synchronized_callback(
        self,
        image_msg,
        entity_msg,
        human_msg,
        human_detected_msg,
        tips_identities_msg,
        tips_embeddings_msg,
        tips_patch_matches_msg,
    ):
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

                # -- Entities (existing detector) ----------------------------
                entities_info = "List of entities detected in this frame (make reference to these exact bounding boxes):\n"

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
                        entities_info += f"- Entity ID: {entity.track_id}, Label: {entity.label}, inside bbox: {x_min}, {y_min}, {x_max}, {y_max}\n"

                # -- Humans (Description) ----------------------------
                humans_info = "Array of human characteristics related to user in this frame:\n"

                if not human_msg.persons:
                    humans_info += "No humans detected in this frame.\n"
                else:
                    for person in human_msg.persons:
                        # Build phrase with id, voice id, gender (from SoftBiometrics), engagement level.
                        humans_info += (
                            f"- ID: {person.id}"
                            f", Voice ID: {person.voice_id}"
                            f", Gender: {person.anonymized_speech.gender}"
                            f" (confidence: {person.anonymized_speech.gender_confidence:.2f})"
                            f", Engagement Level: {person.engagement_status.level}\n"
                        )

                # -- Humans (Description) ----------------------------
                human_detected_info = "List of humans detected in this frame (make reference to these exact bounding boxes):\n"

                if not human_detected_msg.entity_array:
                    human_detected_info += "No humans detected in this frame.\n"
                else:
                    for entity in human_detected_msg.entity_array:
                        bbox = entity.bbox_xyxy

                        # Denormalize bboxes
                        x_min = int(bbox.xmin * pixels_width)
                        y_min = int(bbox.ymin * pixels_height)
                        x_max = int(bbox.xmax * pixels_width)
                        y_max = int(bbox.ymax * pixels_height)

                        # Build phrase with id, label and bounding box (absolute pixel coords).
                        human_detected_info += f"- Human ID: {entity.track_id}, Label: {entity.label}, inside bbox: {x_min}, {y_min}, {x_max}, {y_max}\n"       
                # -- TIPS perception data ------------------------------------
                tips_info = self._build_tips_info(
                    tips_identities_msg,
                    tips_embeddings_msg,
                    tips_patch_matches_msg,
                    pixels_width,
                    pixels_height,
                )

                self.get_logger().info(
                    f"TIPS summary built: "
                    f"{len(tips_identities_msg.identities)} identities, "
                    f"{len(tips_embeddings_msg.embeddings)} embeddings, "
                    f"{len(tips_patch_matches_msg.matches)} patch matches."
                )

                # ----------------------------------------------------------------
                # 4. Build the Scene Graph prompt for Qwen3.6-27b
                # ----------------------------------------------------------------
                # Provide the VLM a prompt that gives the json schema
                # definition to follow strictly.
                # ----------------------------------------------------------------

                system_msg = (
                    "You are a visual perception module on a mobile robot. "
                    "You receive a camera frame together with structured sensor data "
                    "including standard entity detections and rich TIPS perceptual data "
                    "(stable object re-ID, visual embeddings, and text-query patch matches). "
                    "Your job is to analyse the image carefully and call the "
                    "create_scene_graph tool with a complete, accurate scene graph. "
                    "Prioritise what you get from the message data and confirm the relationships with the image; "
                    "Follow a json schema as output."
                )

                bb_prompt = f"""\
                Analyse the camera frame and the sensor data below, then call `create_scene_graph`.

                --- SENSOR DATA (IMAGE: {pixels_width}x{pixels_height} px) ---
                {entities_info}
                --------------------------------------------------------------

                --- HUMAN DATA (IMAGE: {pixels_width}x{pixels_height} px)  ---
                {human_detected_info}
                --------------------------------------------------------------

                --- HUMAN CHARACTERISTICS DATA ---
                {humans_info}
                ----------------------------------

                --- TIPS PERCEPTUAL DATA ---
                The TIPS block below comes from three complementary perception streams:
                  * object_identities : stable cross-frame re-ID (object_id persists across tracking resets).
                  * embeddings        : TIPS v2 ViT features — use category & bbox for grounding.
                  * patch_matches     : each entry names the text query it matched best and the
                                       cosine similarity score; use this to infer object semantics
                                       and fine-grained spatial location.
                  * contour_px        : when present, a sub-sampled polygon (absolute pixel coords)
                                       of the segmentation mask; use it to refine the entity's
                                       exact shape and spatial relationships with neighbouring objects.

                {tips_info}
                ------------------------------------------------------------

                Use the raw image just as a reference for scene graph generation output.
                Your goal is to generate a comprehensive, physically-grounded Scene Graph.
                The output must serve as a deterministic spatial and semantic map for a downstream LLM decision-making agent designed for social robotics and human-robot interaction.

                ------------------------------------------------------------------------
                ALLOWED STATES
                ------------------------------------------------------------------------
                [Object/Inanimate States]: open, closed, empty, full, dirty, clean, hot, cold, turned_on, turned_off, stable, unstable, broken
                [Human Pose States — pick exactly one per human entity]: standing, sitting, walking, pointing, raising_right_hand, raising_left_hand, waving
                [Human Activity States — optional, combine with pose]: reaching, looking_at, interacting, neutral, gesturing
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
                3. Human Pose Classification: For every human entity you MUST assign exactly one pose from the Human Pose States list (standing, sitting, walking, pointing, raising_right_hand, raising_left_hand, waving). Use the bounding box from /humans/detected together with the image to determine the correct pose. You may additionally add one or more Human Activity States.
                4. Spatial & Relative Relationships: Deduce precise relative positions. If Object A is to the left of Object B from the camera perspective, log [A -> on_the_left_of -> B]. If Bounding Box data is deducible, ensure relationships strictly mirror the spatial vectors.
                5. TIPS data usage: Cross-reference entity identities (object_id from object_identities) with the detector track_id to confirm persistent identities across frames. Use patch_match cosine scores and matched queries to refine semantic labels and states. Use contour_px (when present) to sharpen occlusion and proximity relationships between overlapping entities.
                6. JSON Formatting: The final output must be a single, valid JSON object starting with {{ and ending with }}. Do not include any markdown block formatting (like ```json) around the JSON.
                7. Reasoning: If you must reason or explain, do it in a <think>...</think> block at the very beginning of your response, or do it as plain text before the JSON block. Do not include any text, reasoning, or explanations after the closing brace }} of the JSON block.

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
                    self.get_logger().warning(
                        f"VLM returned a {type(response_data).__name__} instead of a dict "
                        f"(likely a bare bounding-box array leaked before the scene graph). "
                        f"Skipping frame {self.counter_}. RAW (first 300): {str(response)[:300]!r}"
                    )
                    return

                # 7. Normalise tool output -> pipeline schema.
                #    The simplified tool schema uses flat fields (box_2d at entity top-level,
                #    entity_type instead of type) to avoid Groq JSON Schema limitations.
                #    Convert back to the nested format expected by the rest of the pipeline.
                for entity in response_data.get("entities", []):
                    # box_2d: flat -> nested in spatial_info
                    if "box_2d" in entity and "spatial_info" not in entity:
                        entity["spatial_info"] = {"box_2d": entity.pop("box_2d")}
                    # entity_type -> type
                    if "entity_type" in entity and "type" not in entity:
                        entity["type"] = entity.pop("entity_type")
                    # action_description: "none"/"" -> None
                    ad = entity.get("action_description", "")
                    if ad in ("", "none", "None", "null", "N/A"):
                        entity["action_description"] = None


                # Save the Scene Graph JSON metadata
                json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OutputData/Scene_Graph_json")
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
