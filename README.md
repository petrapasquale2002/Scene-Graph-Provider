# Scene Graph Provider (GroqQwen Branch)

[![ROS / ROS2](https://img.shields.io/badge/Framework-ROS2%20Kilted-0366d6)](https://www.ros.org/)
[![VLM](https://img.shields.io/badge/VLM-Qwen3--27b%20%28Groq%29-6f42c1)](#)
[![License](https://img.shields.io/badge/License-MIT-28a745)](LICENSE)

This branch of **Scene Graph Provider** is optimized for real-time semantic and spatial HRI understanding by integrating **TIPS (v2) perceptual streams** with **Groq's Qwen3.6-27b** Vision-Language Model.

The node subscribes to multiple sensor streams, synchronizes them, converts bounding boxes and segmentation contours to absolute coordinates, builds a rich `{tips ...}` descriptor, prompts the VLM, and outputs a dynamic, structured **Scene Graph** mapping entities and their spatial/functional relationships.

---

## 👁️ System Architecture & Workflow

The pipeline is divided into three main phases:

```mermaid
graph TD
    A[Camera Frame] -->|CompressedImage - TRIGGER| OnImage["_on_image()"]

    B["Entity Detections (optional)"] -->|EntityArray| CacheE["deque cache"]
    BH["Human Detections (optional)"] -->|EntityArray| CacheHD["deque cache"]
    C["TIPS Identities (optional)"] -->|TipsObjectIdentityArray| CacheI["deque cache"]
    D["TIPS Embeddings (optional)"] -->|TipsEmbeddingArray| CacheEmb["deque cache"]
    E["TIPS Patch Matches (optional)"] -->|TipsPatchMatchArray| CacheP["deque cache"]
    F["Unified Persons (optional)"] -->|PersonArray| CacheHU["deque cache"]

    OnImage -->|"closest msg within slop=0.5s (or empty default)"| CacheE
    OnImage -->|"closest msg within slop=0.5s (or empty default)"| CacheHD
    OnImage -->|"closest msg within slop=0.5s (or empty default)"| CacheI
    OnImage -->|"closest msg within slop=0.5s (or empty default)"| CacheEmb
    OnImage -->|"closest msg within slop=0.5s (or empty default)"| CacheP
    OnImage -->|"closest msg within slop=0.5s (or empty default)"| CacheHU

    OnImage --> Prompt[Sensor data prompt + absolute coordinates]
    Prompt -->|Image + Text Prompt| VLM[VLM Inference: Qwen3 on Groq]
    VLM -->|JSON Response| Parser[Response Normalization]
    Parser -->|SceneGraph String JSON| Pub[/scene_graph]
    Parser -->|Save file| Viewer[scene_graph_viewer.py]
```

### 1. Multi-Topic Subscription & Synchronization
The camera frame is the **sole trigger** of the processing pipeline. Every time a new frame arrives, the node performs a reception-time lookup over circular caches (one per optional topic) to assemble the best-available data packet for that frame.

> **Synchronization clock**: all caches store messages keyed by `time.monotonic()` (wall-clock reception time), not the message header timestamp. This makes the matching robust regardless of rosbag playback speed or VLM inference delay: messages that arrive at the node at the same wall-clock instant are considered temporally consistent, regardless of their original recording timestamp.

**Required topic (always published):**
* **Camera stream** (`/camera/image_raw/compressed`): Compressed JPEG frames. Every incoming frame fires `_on_image()`.

**Optional topics (cache-based, may have gaps):**
* **Entities detected** (`/entities/detected`): Standard 2D bounding boxes and tracking IDs for all scene objects. May be absent if no object is in the scene.
* **Humans detected** (`/humans/detected`): 2D bounding boxes and tracking IDs for human bodies specifically. Used for human pose grounding in the VLM prompt. May be absent if no person is visible.
* **Unified persons** (`/humans/unified`): Rich HRI person descriptors including voice ID, engagement status, and soft biometrics (gender, age). May be absent if no person is tracked.
* **TIPS Object Identities** (`/tips/object_identities`): Stable physical identity keys (`object_id`) from the re-ID database. May be absent if no tracked object is visible.
* **TIPS Dense Embeddings** (`/tips/embeddings`): Visual ViT features and bounding boxes. May be absent if no object is currently being tracked.
* **TIPS Patch Matches** (`/tips/patch_matches`): Fine-grained spatial matching of ViT patches against text queries. May be absent if no query matches are found.

Each optional topic stores its messages in a thread-safe `deque(maxlen=10)`. On every frame, `_closest()` retrieves the entry whose reception time is nearest to the current image reception time within a `slop=0.5 s` window. If no match is found, an empty default message is used, so the pipeline **always continues** regardless of which topics are silent.

### 2. Coordinate Conversion & Prompt Construction
Incoming normalized coordinates `[0.0, 1.0]` for bounding boxes are denormalized to absolute pixel coordinates based on the frame resolution. If segmentation masks (contours) are present, they are sub-sampled and formatted as `contour_px=[(x1, y1), ...]` (max 12 points).
All TIPS features are injected as a structured `{tips ...}` payload into the VLM prompt.

### 3. VLM Inference & Scene Graph Generation
The VLM (Qwen3.6-27b via Groq) uses the prompt to output a JSON scene graph containing:
* **Entities:** ID, semantic label, type (`object` | `human` | `structural`), states (including human pose states: `standing`, `sitting`, `walking`, etc.), and `box_2d`.
* **Relationships:** Strictly directed links (e.g. `[cup] --(on_top_of)--> [table]`).

---

## 📊 Topic Mapping (I/O Interfaces)

| Topic Name | Message Type | Direction | Required | Description |
| :--- | :--- | :--- | :---: | :--- |
| `/camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | **Input** | ✅ Yes | Compressed raw camera frame (BEST_EFFORT QoS). **Primary trigger** — every frame fires `_on_image()`. |
| `/entities/detected` | `hri_msgs/msg/EntityArray` | **Input** | ⚪ Optional | Standard 2D bounding boxes and tracking IDs for all scene objects. Empty `EntityArray` used as fallback. |
| `/humans/detected` | `hri_msgs/msg/EntityArray` | **Input** | ⚪ Optional | 2D bounding boxes for human bodies. Used for human pose grounding. Empty `EntityArray` used as fallback. |
| `/humans/unified` | `hri_msgs/msg/PersonArray` | **Input** | ⚪ Optional | Rich HRI person descriptors: voice ID, engagement status, soft biometrics. Empty `PersonArray` used as fallback. |
| `/tips/object_identities` | `eut_scene_msgs/msg/TipsObjectIdentityArray` | **Input** | ⚪ Optional | Re-ID database keys with stable tracked IDs. May be absent if no trackable object is visible. |
| `/tips/embeddings` | `eut_scene_msgs/msg/TipsEmbeddingArray` | **Input** | ⚪ Optional | Dense ViT visual embeddings. May be absent if no object is currently being tracked. |
| `/tips/patch_matches` | `eut_scene_msgs/msg/TipsPatchMatchArray` | **Input** | ⚪ Optional | Text-query spatial patch matching responses. May be absent if no query yields a match. |
| `/scene_graph` | `std_msgs/msg/String` | **Output** | — | Serialized JSON containing the generated Scene Graph metadata. |

---

## 🛠️ Environment Setup & Usage Guide

Follow these steps to build the custom message interfaces, play the dataset, launch the VLM client node, and view the live graph.

### 1. Compilation
Make sure your ROS workspace contains the message definition packages (`hri_msgs` and `eut_scene_msgs`), then build:
```bash
cd ~/ros2_ws
source /opt/ros/kilted/setup.bash
colcon build --packages-select hri_msgs eut_scene_msgs VLM_node
source install/setup.bash
```

### 2. Playing the Dataset (Terminal 1)
To publish the synchronized sensor streams from the bag:
```bash
cd ~/Eurecat_sample_rosbag_fortis/rosbag_eurecat_lab
source /opt/ros/kilted/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 bag play -l table_eurecat_lab/
```

### 3. Launching the VLM Node (Terminal 2)
Activate the virtual environment containing the VLM API dependencies and launch:
```bash
cd ~/ros2_ws/src/VLM_node
source bin/activate
source /opt/ros/kilted/setup.bash
source ~/ros2_ws/install/setup.bash
python3 MultiTopicSubVLM.py
```

### 4. Visualizing the Scene Graph (Terminal 3)
Run the live web viewer, which polls the generated output JSON files and displays them in a dynamic network graph:
```bash
cd ~/ros2_ws/src/VLM_node
source bin/activate
python3 scene_graph_viewer.py
```
Open [http://localhost:8765](http://localhost:8765) in your web browser. Human nodes will be visually distinguished and include pose indicators (`🧍 standing`, `🪑 sitting`, `🙋 raising_right_hand`, etc.).
