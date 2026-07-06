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
    A[Camera Frame] -->|CompressedImage| Sync[ApproximateTimeSynchronizer]
    B[Entity Detections] -->|EntityArray| Sync
    C[TIPS Identities] -->|TipsObjectIdentityArray| Sync
    D[TIPS Embeddings] -->|TipsEmbeddingArray| Sync
    E[TIPS Patch Matches] -->|TipsPatchMatchArray| Sync
    
    Sync -->|Synchronized Data Packet| Prompt[TIPS prompt + absolute coordinates]
    Prompt -->|Image + Text Prompt| VLM[VLM Inference: Qwen3 on Groq]
    VLM -->|JSON Response| Parser[Response Normalization]
    Parser -->|SceneGraph String JSON| Pub[/scene_graph]
    Parser -->|Save file| Viewer[scene_graph_viewer.py]
```

### 1. Multi-Topic Subscription & Synchronization
To guarantee temporal consistency, the system uses an `ApproximateTimeSynchronizer` (with `slop=0.5` and `BEST_EFFORT` sensor QoS profiles) to align 5 distinct streams into a single logical frame:
* **Camera stream:** Compressed JPEG frames.
* **Entities detected:** Baseline 2D detections.
* **TIPS Object Identities:** Stable physical identity keys (`object_id`) assigned by the tracking/re-ID database.
* **TIPS Dense Embeddings:** Visual features and bounding boxes.
* **TIPS Patch Matches:** Fine-grained spatial matching of ViT patches against text queries.

### 2. Coordinate Conversion & Prompt Construction
Incoming normalized coordinates `[0.0, 1.0]` for bounding boxes are denormalized to absolute pixel coordinates based on the frame resolution. If segmentation masks (contours) are present, they are sub-sampled and formatted as `contour_px=[(x1, y1), ...]` (max 12 points).
All TIPS features are injected as a structured `{tips ...}` payload into the VLM prompt.

### 3. VLM Inference & Scene Graph Generation
The VLM (Qwen3.6-27b via Groq) uses the prompt to output a JSON scene graph containing:
* **Entities:** ID, semantic label, type (`object` | `human` | `structural`), states (including human pose states: `standing`, `sitting`, `walking`, etc.), and `box_2d`.
* **Relationships:** Strictly directed links (e.g. `[cup] --(on_top_of)--> [table]`).

---

## 📊 Topic Mapping (I/O Interfaces)

| Topic Name | Message Type | Direction | Description |
| :--- | :--- | :--- | :--- |
| `/camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | **Input** | Compressed raw camera frame (BEST_EFFORT QoS). |
| `/entities/detected` | `hri_msgs/msg/EntityArray` | **Input** | Standard 2D bounding boxes and tracking IDs. |
| `/tips/object_identities` | `eut_scene_msgs/msg/TipsObjectIdentityArray` | **Input** | Re-ID database keys containing stable tracked IDs. |
| `/tips/embeddings` | `eut_scene_msgs/msg/TipsEmbeddingArray` | **Input** | Dense ViT visual embeddings. |
| `/tips/patch_matches` | `eut_scene_msgs/msg/TipsPatchMatchArray` | **Input** | Text-query spatial patch matching responses. |
| `/scene_graph` | `std_msgs/msg/String` | **Output** | Serialized JSON containing the generated Scene Graph metadata. |

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
