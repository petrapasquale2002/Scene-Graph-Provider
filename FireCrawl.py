import os
import json
import requests
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

# 1. Define function that uses Firecrawl
def firecrawl_scrape(url: str) -> str:
    response = requests.post(
        "https://api.firecrawl.dev/v2/scrape",
        headers={"Authorization": f"Bearer {os.environ.get('FIRECRAWL_API_KEY')}"},
        json={"url": url}
    )
    return json.dumps(response.json())  # FIX: deve restituire una stringa, non un dict

# 2. Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "firecrawl_scrape",
            "description": "Scrape the content of a web page",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to be analyzed"}
                },
                "required": ["url"]
            }
        }
    }
]

# 3. Map function names to actual functions
available_functions = {
    "firecrawl_scrape": firecrawl_scrape  # FIX: mancava questo dizionario
}

