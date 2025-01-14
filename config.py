LOCAL_MODEL_PATH = "models/"
LOCAL_CACHE_PATH = "cache/"
LOCAL_DATA_PATH = "data/"

# Disable all external API calls
ENABLE_EXTERNAL_APIS = False

# Local model configurations
MODEL_CONFIG = {
    "text_generation": "local_llm/",
    "image_generation": "stable_diffusion/",
    "speech": "local_tts/",
    "sentiment": "sentiment_model/"
}

# Database configuration
DATABASE = {
    "path": "conversations.db",
    "backup_path": "backup/conversations.db"
}

# Memory configuration
MEMORY_FILE = "memory.json"
