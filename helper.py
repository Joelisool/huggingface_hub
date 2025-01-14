from config import DATABASE, MEMORY_FILE
import json
import sqlite3
from local_models import LocalModelManager

class Helper:
    def __init__(self):
        self.model_manager = LocalModelManager()
        self.load_memory()

    def load_memory(self):
        try:
            with open(MEMORY_FILE, 'r') as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            self.memory = {"persona": {}, "conversations": [], "context": {}}
            self.save_memory()

    def save_memory(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=4)

    async def process_input(self, user_input):
        # Replace external API calls with local model calls
        response = await self.model_manager.generate_text(user_input)
        return response

    # Add other helper methods here

def adjust_response_to_persona(response, persona_tone, sentiment=None):
    """
    Adjust the response based on the persona tone and optional sentiment.
    The function tailors the response to fit the Echo persona's traits.
    """
    # Check if the response has already been adjusted
    if "_adjusted" in response:
        return response.replace("_adjusted", "")  # Remove adjustment marker and return original response
    
    # Adjust the response based on the persona tone
    adjusted_response = response

    # Persona tone adjustments
    if persona_tone == "friendly":
        adjusted_response = "*smiles warmly* " + adjusted_response
    elif persona_tone == "formal":
        adjusted_response = "Greetings. " + adjusted_response
    elif persona_tone == "casual":
        adjusted_response = "Hey! " + adjusted_response
    elif persona_tone == "empathetic":
        adjusted_response = "*nods understandingly* " + adjusted_response
    elif persona_tone == "supportive":
        adjusted_response = "*gently* " + adjusted_response
    elif persona_tone == "inquisitive":
        adjusted_response = "*curiously* " + adjusted_response

    # Adjust based on sentiment (if available)
    if sentiment:
        if sentiment == "negative":
            adjusted_response = "*comforting* " + adjusted_response
        elif sentiment == "positive":
            adjusted_response = "*enthusiastically* " + adjusted_response

    # Add an adjustment marker to prevent re-adjustment
    adjusted_response += " _adjusted"

    return adjusted_response
