import os
import json
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from litellm import completion
from src.llm_client import LLMClient

load_dotenv(dotenv_path=Path(__file__).resolve().parent / "NebiusAPI.env")


"""
LLM client used for testing Json file comprehension. This model must return a Description of the scene in natural language while referring to the scene graph json file.
The output must be like a list of affirmations. For example:
- The bowl is on the table.
- The person is sitting on the chair.
- The person is pointing at the pot.
It subscribes to the output of node MultiTopicSub.py, reads its output and elaborates an assistance decision
"""

class MinimalSubscriber(Node):

    def __init__(self):

        super().__init__('LLM_sub')

        self.subscription = self.create_subscription(
            String,
            '/scene_graph',
            self.scene_graph_callback,
            10
        )
        self.subscription  # previene warning per variabile non utilizzata

        # Configura il client LLM una sola volta all'avvio del nodo
        model_parameters = self.get_model_config()
        self.client = LLMClient(**model_parameters)

        self.system_message = (
            "You are a helpful assistant that describes the scene based on the provided scene graph JSON data. "
            "The JSON contains information about detected entities, their labels, bounding boxes, and relationships between them. "
            "Build a description of the scene in natural language while referring to the scene graph json file. "
            "The output must be like a list of affirmations. For example:\n"
            "- The bowl is on the table.\n"
            "- The person is sitting on the chair.\n"
            "- The person is pointing at the pot.\n"
            "Moreover, since your task is also to take decision given the context, you must propose an action that "
            "fit the situation and recognise when the operator is in need of something and take care of him."
        )

    def get_model_config(self):
        return {
            "model_name": "groq/llama-3.3-70b-versatile",
            'temperature': 0.3,
            'max_tokens': 2048,
            'top_p': 0.9
        }

    def scene_graph_callback(self, msg):
        self.get_logger().info("Ricevuto un nuovo Scene Graph. Elaborazione con LLM...")
        try:
            # Converte la stringa JSON in un dizionario Python
            data = json.loads(msg.data)
            
            # Estrae la chiave 'scene_graph' se presente, altrimenti usa l'intero dizionario
            scene_graph = data.get("scene_graph", data)

            # Costruisce il messaggio utente con i dati correnti
            user_message = (
                f"Here is the scene graph data: {json.dumps(scene_graph)}. "
                "Please provide a description of the scene based on this data, it must be just a list of affirmations, nothing more. "
                "Then, you must provide an action, like 'Pick up the object X' or similar, as a possible response. "
                "The three actions possible are PICK, PLACE, NAVIGATE. The action are focused on assisting the human."
            )

            # Esegue la chiamata all'LLM
            response = self.client(
                system_message=self.system_message,
                user_message=user_message
            )

            self.get_logger().info("Risposta ricevuta dall'LLM:")
            print("\n" + "="*50)
            print(response)
            print("="*50 + "\n")

        except Exception as e:
            self.get_logger().error(f"Errore durante l'elaborazione del Scene Graph: {str(e)}")


def main(args=None):
    # 1. Inizializza le librerie client ROS2
    rclpy.init(args=args)
    
    # 2. Crea un'istanza del nodo (la tua classe che eredita da Node)
    minimal_subscriber = MinimalSubscriber()

    try:
        # 3. Avvia lo "spin" del nodo.
        # Questa funzione blocca lo script in un ciclo infinito e rimane in attesa di messaggi.
        # Quando un messaggio arriva sul topic '/scene_graph', ROS2 chiama automaticamente 'scene_graph_callback'.
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # 4. Spegnimento pulito: distrugge il nodo e chiude il contesto ROS2
        minimal_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


    
if __name__ == "__main__":
    main()
    