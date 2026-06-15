# Scene Graph Provider

[![ROS / ROS2](https://img.shields.io/badge/Framework-ROS%20%2F%20ROS2-0366d6)](https://www.ros.org/)
[![VLM](https://img.shields.io/badge/AI-Vision--Language%20Model%20%28VLM%29-6f42c1)](#)
[![License](https://img.shields.io/badge/License-MIT-28a745)](LICENSE)

**Scene Graph Provider** is a software component designed as a processing node for robotic ecosystems or event-driven architectures. Its primary goal is to provide real-time semantic and spatial understanding of environments. 

The system subscribes to incoming sensory data streams, processes the visual frames using a **Vision-Language Model (VLM)**, and outputs a dynamic, structured **Scene Graph** mapping the detected objects and their mutual relationships.

---

## 👁️ System Architecture & Workflow

The internal data flow follows an asynchronous, event-driven pipeline divided into three main phases:

### 1. Topic Subscription (Data Ingestion)
The module actively listens to specific communication channels (such as ROS/ROS2 topics or distributed video streams). It continuously captures incoming messages, which typically include:
* Raw video frames or images from onboard cameras (`sensor_msgs/Image`).
* Any associated synchronization metadata or spatial telemetry.

### 2. Vision-Language Model Analysis (VLM Inference)
Whenever a new frame is received from the subscribed topics, it is forwarded to the **VLM** inference engine (e.g., CLIP, LLaVA, BLIP, or similar models configured in the environment). 
The model performs an integrated visual-textual analysis:
* **Object Detection & Grounding:** It identifies objects within the scene and extracts their semantic boundaries.
* **Contextual Understanding:** It leverages the "Language" capability of the model to interpret complex properties that traditional computer vision algorithms struggle to capture (e.g., the state of an object, room context).

### 3. Scene Graph Generation & Publication
The VLM's output is structured into a graph data structure (Scene Graph):
* **Nodes:** Represent detected entities (e.g., `table`, `chair`, `person`) enriched with attributes (color, category, spatial coordinates).
* **Edges (Relations):** Represent spatial or functional connections between nodes (e.g., `[table] --(supports)--> [cup]`, `[chair] --(next to)--> [table]`).

The generated graph is serialized (e.g., into JSON or a custom message format) and **published to a dedicated output topic**, making it available for high-level modules like motion planners, navigation stacks, or decision-making systems.

---

## 📊 Topic Mapping (I/O Interfaces)

The standard communication interfaces utilized by this component are detailed below:

| Topic Name | Message Type / Structure | Direction | Description |
| :--- | :--- | :--- | :--- |
| `/camera/rgb/image_raw` | `sensor_msgs/Image` | **Input** (Subscription) | Video stream coming from the camera sensor or simulation environment. |
| `/scene_graph` | `custom_interfaces/SceneGraph` | **Output** (Publication) | Structured graph format containing the nodes and relations of the current scene. |

---

## 🛠️ Prerequisites & Requirements

* **Python** 3.8+ or **C++17** (depending on the core engine implementation)
* **Middleware:** ROS / ROS2 (recommended) or an equivalent message broker (MQTT/ZeroMQ).
* **AI Backend:** Frameworks for VLM execution (e.g., `torch`, Hugging Face `transformers`, or external API integrations/ONNX for hardware acceleration).

---

## 🚀 Environment Setup & Usage Guide

Below is the complete sequence of terminal commands required to clone the repository, set up the dual-layer environment (Python Virtual Environment + ROS Environment), compile the package, play your local dataset, and launch the core node.

```bash
# ==============================================================================
# SECTION 1: CLONING AND ENVIRONMENT SETUP
# ==============================================================================

# 1. Clone the repository into your workspace
git clone [https://github.com/petrapasquale2002/Scene-Graph-Provider.git](https://github.com/petrapasquale2002/Scene-Graph-Provider.git)
cd Scene-Graph-Provider

# 2. Create and activate a Python virtual environment to isolate VLM dependencies
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install required AI/VLM dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional for ROS/ROS2 workspaces) Compile your custom workspace packages
# Replace <distro> with your active version (e.g., humble, foxy, noetic)
source /opt/ros/<distro>/setup.bash
colcon build --packages-select scene_graph_provider
source install/setup.bash


# ==============================================================================
# SECTION 2: RUNNING THE PIPELINE (RUN IN TERMINAL 1)
# ==============================================================================
# Open a completely new terminal window to play your local ROS bag file.

# 1. Source your global ROS environment
source /opt/ros/<distro>/setup.bash

# 2. Source your local workspace package to expose custom message interfaces
source /path/to/Scene-Graph-Provider/install/setup.bash

# 3. Stream the sensory data from your local computer
# For ROS2 bag structures (.db3 / metadata.yaml directories):
ros2 bag play /path/to/your/local/directory_or_file.db3

# OR if you are using legacy ROS1 bag structures:
# rosbag play /path/to/your/local/file.bag


# ==============================================================================
# SECTION 3: EXECUTING THE PYTHON NODE (RUN IN TERMINAL 2)
# ==============================================================================
# Open a second terminal window to run the VLM processing pipeline.

# 1. Navigate directly to your project repository path
cd /path/to/Scene-Graph-Provider

# 2. Activate the Python virtual environment (loads Torch, Transformers, etc.)
source venv/bin/activate

# 3. Source your global ROS environment dependency layer
source /opt/ros/<distro>/setup.bash

# 4. Source your compiled local workspace definitions
source install/setup.bash

# 5. Execute the main Python subscriber & VLM inference script
python3 src/scene_graph_provider_node.py --model_path /path/to/vlm/weights --config config/params.yaml
