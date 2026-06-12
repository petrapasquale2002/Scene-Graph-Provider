import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from litellm import completion

load_dotenv(dotenv_path=Path(__file__).resolve().parent / "NebiusAPI.env")


"""
LLM client used for testing Json file comprehension. This model must return a Description of the scene in natural language while referring to the scene graph json file.
The output must be like a list of affirmations. For example:
- The bowl is on the table.
- The person is sitting on the chair.
- The person is pointing at the pot.
"""

def test_groq():
    return {
        "model_name": "groq/llama-3.3-70b-versatile",
        'temperature': 0.3,
        'max_tokens': 2048,
        'top_p': 0.9
    }
    

def test_nebius():
    return {
        "model_name": "nebius/meta-llama/Llama-3.3-70B-Instruct",
        'temperature': 0.3,
        'max_tokens': 2048,
        'top_p': 0.9,
        'logprobs': False,
        'full_content': False
    }

if __name__ == "__main__":

    from src.llm_client import LLMClient
    
    use_nebius = True
    use_groq = False

    if use_nebius:
        model_parameters = test_nebius()
    elif use_groq:
        model_parameters = test_groq()

    try:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_graph_6.json")
        with open(json_path, "r") as json_file:
            scene_graph = json.load(json_file)
    except FileNotFoundError:
        print("test_scene_graph.json file not found. Please ensure the file exists in the current directory.")
        exit(1)
    system_message = "You are a helpful assistant that describes the scene based on the provided scene graph JSON data. The JSON contains information about detected entities, their labels, bounding boxes, and relationships between them. Build a description of the scene in natural language while referring to the scene graph json file. The output must be like a list of affirmations. For example: - The bowl is on the table.- The person is sitting on the chair.- The person is pointing at the pot."
    user_message = f"Here is the scene graph data: {json.dumps(scene_graph)}. Please provide a description of the scene based on this data."

    client = LLMClient(
        **model_parameters
    )

    response = client(
        system_message=system_message,
        user_message=user_message
    )

    print(response)