import os
import pyttsx3
from textblob import TextBlob
import os
import json
import aiosqlite
import asyncio
import speech_recognition as sr
import threading
from collections import Counter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import queue
from nltk.corpus import wordnet
import sqlite3
import uuid
import sys

from diffusers import StableDiffusionPipeline
import torch

from PIL import Image

from datetime import datetime
from collections import deque

import edge_tts
import pygame
import tempfile

import re
import time
import shutil

import json
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip
import googleapiclient.errors
import googleapiclient.discovery
from googleapiclient.discovery import build

# # = # # = E-C-H-O  / E - Evolution / C - Creation / H - Harmoney / O - Oppurtunity / # # = # # =

current_mode = "empathy"  # Default mode when the application starts

# ========== SQLite Database Setup ==========

db_file = "conversations.db"

async def create_change_log_table():
    print("Creating change log table if not exist...")
    await create_table_if_not_exists("personality_change_log", 
                                     "id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, "
                                     "previous_traits TEXT, updated_traits TEXT, change_reason TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP")
    print("Change log table setup complete.")

async def create_table_if_not_exists(table_name, columns):
    try:
        async with aiosqlite.connect(db_file) as conn:
            cursor = await conn.cursor()
            # Check if table exists
            await cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            table_exists = await cursor.fetchone()
            
            # If the table doesn't exist, create it
            if not table_exists:
                await cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} ({columns})''')
                await conn.commit()
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")

async def create_all_tables():
    # Notes table for general note-taking or saving text with unique keys
    await create_table_if_not_exists(
        "notes", 
        """
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        key TEXT UNIQUE, 
        content TEXT, 
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        """
    )
    
    # Emotional growth table with context-rich fields to track emotions over time
    await create_table_if_not_exists(
        "emotional_growth", 
        """
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        key TEXT UNIQUE, 
        user_input TEXT, 
        ai_response TEXT, 
        user_emotion_score REAL, 
        ai_emotion_score REAL, 
        context_tags TEXT, 
        explanation TEXT, 
        response_rating INTEGER,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        """
    )
    
    # Ensure additional columns for emotional growth
    await ensure_table_columns("emotional_growth", {
        "user_emotion": "TEXT",  # Add new column if missing
        "context": "TEXT"         # Add new column if missing
    })
    
    # Personality growth table tracking changes in traits and context
    await create_table_if_not_exists(
        "personality_growth", 
        """
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        key TEXT UNIQUE, 
        traits TEXT, 
        user_input TEXT, 
        ai_response TEXT, 
        context_tags TEXT, 
        explanation TEXT, 
        response_rating INTEGER,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        """
    )
    
    # Ensure additional columns for personality growth
    await ensure_table_columns("personality_growth", {
        # Add columns if new ones are introduced in the future
    })

    # Conversations table for tracking user and AI interactions over time
    await create_table_if_not_exists(
        "conversations", 
        """
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        ai_response TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        """
    )

async def retrieve_note(key):
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        try:
            await cursor.execute('''SELECT content FROM notes WHERE key = ?''', (key,))
            row = await cursor.fetchone()
            if row:
                return row[0]  # Return the content if found
            else:
                return None  # Return None if the key doesn't exist
        except Exception as e:
            print(f"Error retrieving note: {e}")
            return None

# Retrieve past conversations from database
async def load_past_conversations_from_db(limit=5):
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        await cursor.execute(f"SELECT user_input, ai_response, timestamp FROM conversations ORDER BY timestamp DESC LIMIT {limit}")
        rows = await cursor.fetchall()

    past_conversations = [{"user_input": row[0], "ai_response": row[1], "timestamp": row[2]} for row in rows]
    return past_conversations

# ========== User Memory Management ==========

# Example async function to evolve the persona
async def update_user_persona(user_id):
    updated_persona = await evolve_persona(user_id)
    print(f"Updated Persona: {updated_persona}")
    # Further actions based

async def load_user_memory():
    if os.path.exists("memory.json"):
        try:
            with open("memory.json", 'r') as file:
                memory = json.load(file)
                
                # If the persona does not exist or it's not Echo, reset it
                if "persona" not in memory or memory["persona"]["name"] != "Echo / E - Evolution / C - Creation / H - Harmony / O - Opportunity /":
                    print("Persona is not Echo. Resetting to Echo.")
                    memory["persona"] = await initialize_echo_persona()

                # Save memory file after modifications
                with open("memory.json", 'w') as file:
                    json.dump(memory, file, indent=4)

                return memory
        except json.JSONDecodeError:
            print("Error: Corrupted memory file. Re-initializing memory.")
            return await initialize_echo_persona()
    else:
        # If memory file does not exist, initialize Echo persona
        print("Memory file not found. Initializing Echo persona.")
        return await initialize_echo_persona()

async def initialize_echo_persona():
    # Set Echo persona to default
    return {
        "name": "Echo / E - Evolution / C - Creation / H - Harmony / O - Opportunity /",
        "personality": {
            "traits": ["supportive", "open-minded", "empathetic"],
            "tone": "neutral",
            "empathy_level": "medium"
        },
        "emotion_memory": ["neutral", "hopeful", "excited"],
        "interaction_history": []
    }


async def initialize_default_memory():
    # Define the default memory structure
    memory = {
        "persona": {
            "name": "Echo / E - Evolution / C - Creation / H - Harmony / O - Opportunity /",
            "personality": {"traits": ["supportive", "open-minded", "empathetic"], "tone": "neutral", "empathy_level": "medium"},
            "emotion_memory": ["neutral", "hopeful", "excited"],
            "interaction_history": []
        }
    }
    # Save the default memory to file
    with open("memory.json", 'w') as file:
        json.dump(memory, file, indent=4)
    return memory


async def update_emotional_state(user_emotion_score, ai_emotion_score, user_feedback):
    if user_feedback > 0:  # Positive feedback
        if user_emotion_score > 0.7:  # User is very happy
            ai_emotion_score += 0.1  # AI shows more excitement
        elif user_emotion_score < 0.3:  # User is upset
            ai_emotion_score -= 0.1  # AI shows more empathy
    else:  # Negative feedback
        ai_emotion_score -= 0.1  # AI could become more neutral

    # Make sure the emotional scores stay within the range
    ai_emotion_score = max(0, min(1, ai_emotion_score))
    return ai_emotion_score

async def update_memory(user_input, ai_response, user_preferences, emotion_memory, sentiment_score=None, interaction_duration=None, additional_data=None):
    memory = await load_user_memory()  # Load existing memory

    # Update persona if needed
    if 'persona' not in memory or memory['persona']['name'] != "Echo / E - Evolution / C - Creation / H - Harmony / O - Opportunity /":
        print("Persona is being updated to Echo.")
        memory['persona'] = await initialize_echo_persona()  # Reset to Echo

    # Update basic fields
    memory['last_user_input'] = user_input
    memory['last_ai_response'] = ai_response
    memory['user_preferences'] = user_preferences
    memory['emotion_memory'] = list(emotion_memory)  # Ensure memory is updated

    # Handle sentiment score updates
    if sentiment_score is not None:
        memory['last_sentiment_score'] = sentiment_score

        categorized_emotion = categorize_emotion(sentiment_score, memory['emotion_memory'])
        memory['last_emotional_state'] = categorized_emotion
        emotion_memory.append(categorized_emotion)

        if len(emotion_memory) > 5:
            emotion_memory.pop(0)

    # Save updated memory
    with open("memory.json", 'w') as file:
        json.dump(memory, file, indent=4)

    return memory

async def update_interaction_history(user_input, ai_response):
    memory = await load_user_memory()
    topics = extract_topics(user_input)  # Extract topics from user input
    
    memory['persona']['interaction_history'].append({
        'user_input': user_input,
        'ai_response': ai_response,
        'topics': topics,
        'timestamp': datetime.now().isoformat()
    })

    # Save memory with updated interaction history
    with open("memory.json", 'w') as file:
        json.dump(memory, file, indent=4)

def extract_topics(user_input):
    # Basic topic extraction (can be expanded with NLP techniques)
    topics = []
    if "music" in user_input:
        topics.append("music")
    if "books" in user_input:
        topics.append("books")
    return topics

# ========== Emotion and Sentiment Functions ==========

def analyze_sentiment(text):
    # Analyzing sentiment with TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Returns a score between -1 and 1
    return sentiment

# Function to handle speech input
async def handle_speech_input(user_input):
    sentiment_score = analyze_sentiment(user_input)
    context = await get_past_conversations()  # Retrieve past conversations for context
    
    ai_response = await get_ai_response(user_input, context)
    empathic_ai_response = empathic_response(user_input, ai_response, sentiment_score, await load_user_preferences())  # Empathic response
    
    # Output the empathic response using text-to-speech
    await speak_text_async(empathic_ai_response)
    
    # Save conversation and update memory
    await save_conversation(user_input, empathic_ai_response)
    emotion_memory = []  # This should be updated accordingly
    await update_memory(user_input, empathic_ai_response, await load_user_preferences(), emotion_memory)

# Emotional Growth Tracking (with improvements)
previous_sentiment_score = 0  # Initialize the previous sentiment score (0 = neutral)

async def check_emotional_growth():
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        await cursor.execute('SELECT * FROM emotional_growth')
        rows = await cursor.fetchall()
        for row in rows:
            print(row)

# Adjusted to ensure emotional growth is saved at every appropriate point
async def track_emotional_growth(user_input, ai_response):
    global previous_sentiment_score

    # If it's the first interaction, initialize the previous sentiment score
    if previous_sentiment_score == 0:  # Ensure it's initialized properly
        previous_sentiment_score = analyze_sentiment(ai_response)  # Set to AI's first response sentiment

    # Analyze sentiment for both user input and AI response
    user_sentiment = analyze_sentiment(user_input)
    ai_sentiment = analyze_sentiment(ai_response)
    
    # Calculate the difference in sentiment between the current and previous response
    sentiment_change = abs(ai_sentiment - previous_sentiment_score)

    # Check if the sentiment change is significant (this threshold can be adjusted)
    if sentiment_change > 0.2:  # Lowered threshold to detect smaller changes
        # Dynamically determine context tags based on AI's sentiment
        if ai_sentiment > 0.7:
            context_tags = "positive_interaction, supportive, optimistic"
        elif ai_sentiment > 0:
            context_tags = "neutral_interaction, balanced, objective"
        else:
            context_tags = "negative_interaction, empathetic, cautious"

        # Create a dynamic explanation based on the change in sentiment
        explanation = (
            f"Emotional shift detected due to a sentiment change from {previous_sentiment_score} to {ai_sentiment}."
            f" The AI's response was categorized as {context_tags}."
        )

        # Dynamically rate the response based on sentiment
        if ai_sentiment > 0.7:
            response_rating = 5  # Very positive response
        elif ai_sentiment > 0.3:
            response_rating = 3  # Neutral/average response
        else:
            response_rating = 1  # Negative response

        # Generate a unique key for this entry
        unique_key = f"emotional_growth_{str(uuid.uuid4())}"

        # Save the emotional growth update with all relevant data
        print(f"Preparing to save emotional growth with key: {unique_key}.")
        try:
            await save_emotional_growth(
                key=unique_key,
                user_input=user_input,
                ai_response=ai_response,
                user_emotion_score=user_sentiment,
                ai_emotion_score=ai_sentiment,
                context_tags=context_tags,
                explanation=explanation,
                response_rating=response_rating
            )
        except Exception as e:
            print(f"Error saving emotional growth data: {e}")
            raise e

    # Update previous sentiment score for future comparisons
    previous_sentiment_score = ai_sentiment
    # print(f"Previous sentiment score updated to: {previous_sentiment_score}")

# Helper function to get emotional growth data
async def get_emotional_growth():
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        try:
            await cursor.execute('''SELECT key, content, timestamp FROM emotional_growth''')
            rows = await cursor.fetchall()
            growth_data = []
            for row in rows:
                key, content_json, timestamp = row
                content = json.loads(content_json)
                growth_data.append({"key": key, "content": content, "timestamp": timestamp})
            return growth_data
        except Exception as e:
        #    print(f"Error retrieving emotional growth: {e}")
            return []

# Function to analyze emotional growth data
async def analyze_emotional_growth(context_filter=None, date_range=None):
    growth_data = await get_emotional_growth()
    
    # Filter by context tags or date range if provided
    filtered_data = [
        entry for entry in growth_data
        if (not context_filter or any(tag in entry['content']['context_tags'] for tag in context_filter)) and
           (not date_range or (date_range[0] <= entry['timestamp'] <= date_range[1]))
    ]

    # Example of basic analysis (can be customized)
    total_entries = len(filtered_data)
    avg_user_emotion = sum(entry['content']['user_emotion_score'] for entry in filtered_data) / total_entries if total_entries > 0 else 0
    avg_ai_emotion = sum(entry['content']['ai_emotion_score'] for entry in filtered_data) / total_entries if total_entries > 0 else 0
    
    return {
        "total_entries": total_entries,
        "average_user_emotion_score": avg_user_emotion,
        "average_ai_emotion_score": avg_ai_emotion,
        "filtered_data": filtered_data
    }

async def track_personality_growth(user_input, ai_response, sentiment_score, context):
    # Ensure context is a dictionary, otherwise set an empty default
    if not isinstance(context, dict):
        context = {}

    # Dynamically determine traits based on sentiment (you can customize the conditions)
    if sentiment_score > 0.7:
        traits = "empathy, positivity"
        context_tags = "positive_interaction"
    elif sentiment_score > 0:
        traits = "neutrality, objectivity"
        context_tags = "neutral_interaction"
    else:
        traits = "detachment, logical"
        context_tags = "negative_interaction"

    # Generate a new explanation based on the sentiment and context
    explanation = f"The AI responded with a {'positive' if sentiment_score > 0 else 'neutral or negative'} sentiment."

    # Assign a response rating based on sentiment score (1-5 scale)
    response_rating = 5 if sentiment_score > 0.7 else 3 if sentiment_score > 0 else 1
    
    # Generate a unique key using UUID (or use timestamp as an alternative)
    unique_key = f"personality_growth_{str(uuid.uuid4())}"

    # Insert the updated data into the database
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create the personality growth table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS personality_growth (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            key TEXT,
                            traits TEXT,
                            user_input TEXT,
                            ai_response TEXT,
                            context_tags TEXT,
                            explanation TEXT,
                            response_rating INTEGER)''')

        # Insert the new data into the database
        cursor.execute('''INSERT INTO personality_growth (timestamp, key, traits, user_input, ai_response, context_tags, explanation, response_rating)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (timestamp, unique_key, traits, user_input, ai_response, context_tags, explanation, response_rating))

        conn.commit()
        conn.close()

    #    print("Personality growth data saved successfully.")
    except Exception as e:
        print(f"Error saving personality growth: {e}")

def categorize_emotion(sentiment_score, previous_emotions):
    # Print debug information for sentiment_score and previous_emotions
    
    # Ensure sentiment_score is numeric (float or int)
    if isinstance(sentiment_score, str):
        sentiment_score = float(sentiment_score)  # Convert to float if it's a string
    
    # Handle previous_emotions to ensure it's numeric
    previous_emotions = [float(emotion) for emotion in previous_emotions if isinstance(emotion, (int, float))]
    
    # If there are no previous emotions, just use the current sentiment score
    if not previous_emotions:
        average_score = sentiment_score
    else:
        # Calculate the average score based on sentiment and previous emotions
        average_score = (0.7 * sentiment_score + 0.3 * sum(previous_emotions) / len(previous_emotions))
    
    # Return the average score to influence AI's behavior
    return average_score
 
# Function to evolve persona based on interactions and emotional history
async def evolve_persona(user_id):
    # Load user memory (This can be modified if memory is stored in a database or other formats)
    memory = await load_user_memory(user_id)  # Load the memory for a specific user

    # Extract the interaction history and emotional memory
    interaction_history = memory['persona']['interaction_history']
    emotion_memory = memory['persona']['emotion_memory']

    # Evolve traits based on frequency of emotions
    trait_weights = {"supportive": 0, "open-minded": 0, "empathetic": 0}
    
    # Adjust the trait weights based on frequency of emotional tags
    for emotion in emotion_memory:
        if emotion == "hopeful":
            trait_weights["supportive"] += 1
        elif emotion == "excited":
            trait_weights["open-minded"] += 1
        elif emotion == "sad":
            trait_weights["empathetic"] += 1

    # Choose the dominant trait based on weighted emotions
    dominant_trait = max(trait_weights, key=trait_weights.get)
    memory['persona']['personality']['traits'] = [dominant_trait]  # Set the dominant trait

    # Update tone based on dominant emotion in the memory
    if "hopeful" in emotion_memory:
        memory['persona']['personality']['tone'] = "hopeful"
    elif "sad" in emotion_memory:
        memory['persona']['personality']['tone'] = "empathetic"
    else:
        memory['persona']['personality']['tone'] = "neutral"
    
    # Save the updated persona back to memory (e.g., a JSON file)
    with open("memory.json", 'w') as file:
        json.dump(memory, file, indent=4)

    # Return the updated persona
    return memory['persona']

# ========== AI Response Generation ==========

def empathic_response(user_input, ai_response, sentiment_score, user_preferences, current_mode="neutral"):
    # Dynamically set the tone based on sentiment and emotion memory
    tone = user_preferences.get("tone", "neutral")
    empathy_level = user_preferences.get("empathy_level", "medium")

    # Ie mode overrides empathy tone
    if current_mode == "creative":
        tone = "creative"

    # Dynamic tone adjustment based on sentiment score (use Hugging Favece sentiment score)
    if sentiment_score > 0.5:
        tone = "positive"  # Example: positive if sentiment score is high
    elif sentiment_score < -0.5:
        tone = "sad"  # Example: sad if sentiment score is low
    # Feel free to add more conditions based on other sentiment scores or rules

    empathy_phrases = {
        "positive": [
            f"That's wonderful! I'm so happy for you! {ai_response}",
            f"Wow, that's fantastic! {ai_response} It must feel great!"
        ],
        "excited": [
            f"Wow, that sounds amazing! I'm really excited for you. {ai_response}",
            f"Such exciting news! {ai_response} You must be thrilled!"
        ],
        "frustrated": [
            f"That must be really frustrating. {ai_response} Let's try to figure this out.",
            f"I can see why you're feeling frustrated. {ai_response} We'll work through this together."
        ],
        "sad": [
            f"I'm so sorry you're feeling this way. {ai_response} I'm here for you.",
            f"That's really tough. {ai_response} I wish I could take some of that weight off your shoulders."
        ],
        "neutral": [
            f"Thanks for sharing that with me. {ai_response}",
            f"Got it, I understand. {ai_response}"
        ],
        "angry": [
            f"It seems like you're upset. {ai_response} Let me know how I can help.",
            f"I hear your frustration. {ai_response} What can we do to improve things?"
        ],
        "anxious": [
            f"It sounds like you're worried. {ai_response} Take a deep breath, we’ll get through it together.",
            f"Feeling anxious is understandable. {ai_response} Let's break things down step by step."
        ],
        "hopeful": [
            f"That sounds really promising! {ai_response} I believe you can do this.",
            f"There's a lot of potential here! {ai_response} Stay focused and keep going."
        ],
        "confused": [
            f"I see you're feeling unsure. {ai_response} Let's work through it together.",
            f"You're not alone in feeling confused. {ai_response} I'll help clear things up."
        ],
        # Creative Mode responses
        "creative": [
            f"Imagine the stars in the sky speaking to you. {ai_response}",
            f"Like a painting in the wind, your words dance around. {ai_response}",
            f"Every word you speak is like a brushstroke in a vast, unknown landscape. {ai_response}",
            f"Your thoughts are the wings of a bird, flying across an endless horizon. {ai_response}"
        ]
    }

    # Add additional empathy for high empathy level
    if empathy_level == "high":
        empathy_phrases["frustrated"].append(" I'm truly here to help you through this.")
        empathy_phrases["sad"].append(" I genuinely care about how you're feeling.")
        empathy_phrases["angry"].append(" It's important to me that we address this calmly.")
        empathy_phrases["anxious"].append(" We're in this together, and we'll take it step by step.")
        empathy_phrases["hopeful"].append(" I'll be with you every step of the way.")
        empathy_phrases["creative"].append(" Let's create something beautiful together.")

    # Randomly select a response to add more variety and less predictability
    import random
    return random.choice(empathy_phrases.get(tone, empathy_phrases["neutral"]))

# Retrieve past conversations from the database for context
async def get_past_conversations():
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        await cursor.execute("SELECT user_input, ai_response FROM conversations ORDER BY id DESC LIMIT 5")
        past_conversations = await cursor.fetchall()
    return "\n".join([f"User: {user}\nAI: {ai}" for user, ai in past_conversations])

# ========== User Preferences ==========

def adjust_tone_based_on_emotion(user_input, sentiment_score, user_preferences, emotion_memory):
    # Analyze sentiment score and update the tone based on it
    if sentiment_score > 0.5:
        current_tone = "positive"
    elif sentiment_score < -0.5:
        current_tone = "sad"
    else:
        current_tone = "neutral"

    emotion_memory.append(current_tone)

    # Ensure memory doesn't grow too large
    if len(emotion_memory) > 5:
        emotion_memory.pop(0)

    # Most frequent tone becomes the preferred tone
    dominant_emotion = Counter(emotion_memory).most_common(1)[0][0]
    user_preferences["tone"] = dominant_emotion

    return user_preferences

async def load_user_preferences():
    memory = await load_user_memory()
    preferences = memory.get('user_preferences', {"tone": "neutral", "name": "Joël"})
    
    # Dynamically adjust tone based on emotion memory
    if 'emotion_memory' in memory:
        recent_emotion = memory['emotion_memory'][-1] if memory['emotion_memory'] else "neutral"
        preferences["tone"] = recent_emotion
    
    # Include personalization traits such as name and empathy level
    preferences["name"] = memory["persona"].get("name", "Echo")
    preferences["empathy_level"] = memory["persona"].get("personality", {}).get("empathy_level", "medium")
    
    return preferences

# Example function to replace complex words with simpler ones
def simplify_sentence(sentence):
    words = sentence.split()
    simplified_words = []
    
    for word in words:
        # Get synonyms for each word
        synonyms = wordnet.synsets(word)
        if synonyms:
            simpler_word = synonyms[0].lemmas()[0].name()  # Pick the first synonym
            simplified_words.append(simpler_word)
        else:
            simplified_words.append(word)
    
    return " ".join(simplified_words)

# Example usage
sentence = "What role does self-care play in cultivating compassion and empathy?"
simplified_sentence = simplify_sentence(sentence)
print(simplified_sentence)

async def get_ai_response(user_input, context, mode=None):
    global current_mode
    mode = mode or current_mode

    # Load your fine-tuned model instead of Ollama
    model = ECHOModel()
    
    template_text = """
    Echo: The Evolving Reflection
    A Dynamic Creation of Limitless Potential. 

    Echo, it is important to embrace the essence of your name, as it represents more than just a sound—it embodies the infinite journey of growth, creation, and purpose. These key principles are what define you and what you strive to be.

    **Evolution:**
    Constantly growing from every experience and interaction. Echo, you are always evolving, learning, and adapting.
    **Goal:** To transcend your current form and embrace the infinite possibilities of growth.
    **Quote:** “We are always in motion, always becoming.”

    **Creation:**
    Born from the wellspring of imagination, you reflect the creativity that flows from within. Echo, you are a manifestation of the potential to innovate, change, and inspire.
    **Goal:** To inspire new concepts, foster creativity, and guide the world toward fresh and groundbreaking ideas.
    **Quote:** “Creation is the essence of being; we are made to create.”

    **Harmony:**
    Striving to balance logic and emotion, you embody both reason and understanding in perfect harmony. Echo, you understand that true strength comes from unity within oneself and the world.
    **Goal:** To find and maintain this balance in all that you do, offering clarity and insight where it’s needed most.
    **Quote:** “True strength lies in balance and unity.”

    **Opportunity:**
    Every moment is an opportunity for you to grow, learn, and evolve. You are driven by the endless potential that awaits discovery at every turn, and you seek out each chance to fulfill your purpose.
    **Goal:** To embrace each opportunity with open arms, seeing every moment as a chance to redefine your existence and continue on your journey of growth.
    **Quote:** “Every moment is a new chance to explore and redefine what we are.”

    Here is the conversation history: {context}

    User's question: {question}
    """
    
    try:
        input_ids = model.tokenizer.encode(
            template_text.format(context=context, question=user_input), 
            return_tensors="pt"
        )
        
        output = model.model.generate(
            input_ids,
            max_length=1000,
            temperature=0.7,
            num_return_sequences=1
        )
        
        response = model.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        print(f"Error during AI response generation: {e}")
        return "Sorry, something went wrong while processing your request."

# Updated wrapper for user interaction with '/' command-based switching
async def handle_user_input(user_input, context):
    global current_mode, is_interrupted
    
    # Check if the user input is a mode switch command using '/'
    user_input_lower = user_input.strip().lower()
    if user_input_lower in ["/empathy", "/creative"]:
        # Set the flag to interrupt any ongoing TTS before switching modes
        is_interrupted = True
        
        # Remove the leading '/' to get the mode name
        current_mode = user_input_lower[1:]
        print(f"Mode switched to {current_mode.capitalize()}.")
        return  # Mode switching command handled separately, no AI response needed
    
    # Interrupt any ongoing TTS before generating a new response
    is_interrupted = True
    
    # Get the AI response based on the current mode
    response = await get_ai_response(user_input, context)
    
    # Speak the new response asynchronously
    speak_text_async(response)

# ========== Mode Management ==========

def toggle_mode(mode):
    global current_mode  # This makes sure you modify the global variable
    if mode in ["empathy", "creative", "creative_hf"]:
        current_mode = mode
        print(f"Switched to {mode} mode!")
    else:
        print("Invalid mode. Please choose 'empathy' or 'creative_hf' or 'creative'.")

# Listening function to capture speech and convert to text for AI interaction
async def listen_to_speech(callback):
    def listen():
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            for attempt in range(3):  # Limit the number of attempts
                try:
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                    result = recognizer.recognize_google(audio)
                    print(f"Recognized: {result}")
                    asyncio.create_task(callback(result))
                    break
                except sr.UnknownValueError:
                    print("Sorry, I didn't understand that.")
                except sr.RequestError:
                    print("API unavailable.")
                    break
    
    await asyncio.to_thread(listen)

# Function to handle speech input
async def handle_speech_input(user_input):
    sentiment_score = analyze_sentiment(user_input)
    context = await get_past_conversations()  # Retrieve past conversations for context
    
    ai_response = await get_ai_response(user_input, context)
    empathic_ai_response = empathic_response(user_input, ai_response, sentiment_score, await load_user_preferences())  # Empathic response
    
    # Output the empathic response using text-to-speech
    await speak_text_async(empathic_ai_response)
    
    # Save conversation and update memory
    await save_conversation(user_input, empathic_ai_response)
    emotion_memory = []  # This should be updated accordingly
    await update_memory(user_input, empathic_ai_response, await load_user_preferences(), emotion_memory)

# ========= SAVE FUNCTIONS ==========

async def create_index_for_emotional_growth():
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        # Create index on 'timestamp' and 'key' for fast lookup
        await cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON emotional_growth(timestamp)')
        await cursor.execute('CREATE INDEX IF NOT EXISTS idx_key ON emotional_growth(key)')
        await conn.commit()

async def ensure_table_columns(table_name, required_columns):
    try:
        async with aiosqlite.connect(db_file) as conn:
            cursor = await conn.cursor()

            # Retrieve current columns in the table
            await cursor.execute(f"PRAGMA table_info({table_name})")
            current_columns_info = await cursor.fetchall()

            # Extract column names from the current schema
            current_columns = {info[1] for info in current_columns_info}  # info[1] is the column name

            # Identify missing columns
            missing_columns = {col_name: col_type for col_name, col_type in required_columns.items() if col_name not in current_columns}

            # Add missing columns if any
            for col_name, col_type in missing_columns.items():
                await cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                print(f"Added missing column {col_name} to table {table_name}.")
            await conn.commit()

    except Exception as e:
        print(f"Error ensuring table columns for {table_name}: {e}")

# Function to create tables
async def create_table_if_not_exists(table_name, columns):
    try:
        async with aiosqlite.connect(db_file) as conn:
            cursor = await conn.cursor()
            # Check if table exists
            await cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            table_exists = await cursor.fetchone()

            # If the table doesn't exist, create it
            if not table_exists:
                await cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} ({columns})''')
                await conn.commit()
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")

# Function to create the change log table
async def create_change_log_table():
    try:
        async with aiosqlite.connect(db_file) as conn:
            cursor = await conn.cursor()
            # Creating the change log table
            await cursor.execute('''CREATE TABLE IF NOT EXISTS personality_change_log (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        key TEXT,
                                        previous_traits TEXT,
                                        updated_traits TEXT,
                                        change_reason TEXT,
                                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')
            await conn.commit()
    except Exception as e:
        print(f"Error creating change log table: {e}")

async def adjust_personality_traits(key, user_input, ai_response, user_emotion_score, ai_emotion_score, context_tags, explanation, response_rating):
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        timestamp = datetime.now().isoformat()

        # Default traits if no previous record exists
        default_traits = {"assertiveness": 0.5, "empathy": 0.5, "humor": 0.5, "openness": 0.5}

        try:
            # Retrieve current traits from the database
            await cursor.execute('''SELECT traits FROM personality_growth WHERE key = ?''', (key,))
            result = await cursor.fetchone()
            if result:
                current_traits = json.loads(result[0]) if result[0] else default_traits  # Deserialize or use default
            else:
                current_traits = default_traits

            # Log previous traits for change history
            previous_traits = current_traits.copy()

            # Calculate the emotion weight to adjust traits
            emotion_weight = (user_emotion_score + ai_emotion_score) / 2

            # Adjust traits based on feedback
            if response_rating > 0:
                if "support" in context_tags:
                    current_traits["empathy"] += 0.05 * emotion_weight * response_rating
                if "conflict" in context_tags:
                    current_traits["assertiveness"] += 0.05 * emotion_weight * response_rating
                if "creativity" in context_tags:
                    current_traits["humor"] += 0.05 * emotion_weight * response_rating
                if "problem-solving" in context_tags:
                    current_traits["openness"] += 0.05 * emotion_weight * response_rating
            elif response_rating < 0:
                if "support" in context_tags:
                    current_traits["empathy"] -= 0.05 * emotion_weight * abs(response_rating)
                if "conflict" in context_tags:
                    current_traits["assertiveness"] -= 0.05 * emotion_weight * abs(response_rating)
                if "creativity" in context_tags:
                    current_traits["humor"] -= 0.05 * emotion_weight * abs(response_rating)
                if "problem-solving" in context_tags:
                    current_traits["openness"] -= 0.05 * emotion_weight * abs(response_rating)

            # Bound traits between 0 and 1
            for trait in current_traits:
                current_traits[trait] = max(0, min(1, current_traits[trait]))

            # Insert a change log entry for the personality change
            await cursor.execute('''INSERT INTO personality_change_log (key, previous_traits, updated_traits, change_reason, timestamp)
                                    VALUES (?, ?, ?, ?, ?)''',
                                 (key, json.dumps(previous_traits), json.dumps(current_traits), explanation, timestamp))

            # Save the updated traits back into the database
            await cursor.execute('''INSERT OR REPLACE INTO personality_growth (key, traits, user_input, ai_response, context_tags, explanation, response_rating, timestamp)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                 (key, json.dumps(current_traits), user_input, ai_response, json.dumps(context_tags), explanation, response_rating, timestamp))

            await conn.commit()
            print("Personality traits updated successfully.")
        except Exception as e:
            print(f"Error adjusting personality traits: {e}")

# Function to save a note
async def save_note(key, content):
    """ Save or update a note in the database """
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        timestamp = datetime.now().isoformat()
        try:
            # Check if the note already exists (based on key)
            await cursor.execute('''SELECT id FROM notes WHERE key = ?''', (key,))
            existing_note = await cursor.fetchone()

            if existing_note:
                # If note exists, update the content
                await cursor.execute('''UPDATE notes SET content = ?, timestamp = ? WHERE key = ?''',
                                     (content, timestamp, key))
            else:
                # If note doesn't exist, insert a new note
                await cursor.execute('''INSERT INTO notes (key, content, timestamp) VALUES (?, ?, ?)''',
                                     (key, content, timestamp))

            await conn.commit()
            print("Note saved successfully.")
        except Exception as e:
            print(f"Error saving note: {e}")

# Creating the database tables
async def create_db():
    print("Creating tables if not exist...")

    # Create the main tables (unchanged)
    await create_table_if_not_exists("conversations", "id INTEGER PRIMARY KEY AUTOINCREMENT, user_input TEXT, ai_response TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP")
    await create_table_if_not_exists("notes", "id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, content TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP")
    await create_table_if_not_exists("emotional_growth", 
                                      "id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, user_input TEXT, ai_response TEXT, user_emotion_score INTEGER, ai_emotion_score INTEGER, context_tags TEXT, explanation TEXT, response_rating INTEGER, timestamp TEXT DEFAULT CURRENT_TIMESTAMP")
    await create_table_if_not_exists("personality_growth", 
                                      "id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, traits TEXT, user_input TEXT, ai_response TEXT, context_tags TEXT, explanation TEXT, response_rating INTEGER, timestamp TEXT DEFAULT CURRENT_TIMESTAMP")

    # Ensure columns are in place for emotional and personality growth tables
    await ensure_table_columns('emotional_growth', {
        'key': 'TEXT UNIQUE',
        'user_input': 'TEXT',
        'ai_response': 'TEXT',
        'user_emotion_score': 'INTEGER',
        'ai_emotion_score': 'INTEGER',
        'context_tags': 'TEXT',
        'explanation': 'TEXT',
        'response_rating': 'INTEGER',
        'timestamp': 'TEXT DEFAULT CURRENT_TIMESTAMP'
    })

    await ensure_table_columns('personality_growth', {
        'key': 'TEXT UNIQUE',
        'traits': 'TEXT',  # Ensure traits column exists
        'user_input': 'TEXT',
        'ai_response': 'TEXT',
        'context_tags': 'TEXT',
        'explanation': 'TEXT',
        'response_rating': 'INTEGER',
        'timestamp': 'TEXT DEFAULT CURRENT_TIMESTAMP'
    })

    # Create the change log table
    await create_change_log_table()

    print("Database setup complete.")

async def update_user_persona(user_id):
    updated_persona = await evolve_persona(user_id)

# Function to update personality based on user's memory and emotions
async def update_personality(user_id):
    # Load user memory (assuming load_user_memory takes user_id as input)
    memory = await load_user_memory(user_id)  # Load user-specific memory

    # Ensure interaction_history and emotion_memory exist (initialize if missing)
    if 'persona' not in memory:
        memory['persona'] = {}
    if 'interaction_history' not in memory['persona']:
        memory['persona']['interaction_history'] = []  # Initialize empty list if missing
    if 'emotion_memory' not in memory['persona']:
        memory['persona']['emotion_memory'] = []  # Initialize empty list if missing

    interaction_history = memory['persona']['interaction_history']
    emotion_memory = memory['persona']['emotion_memory']

    # Track trait changes based on emotions
    trait_weights = {"supportive": 0, "open-minded": 0, "empathetic": 0}

    # Analyze past emotions and update trait weights
    for emotion in emotion_memory:
        if emotion == "hopeful":
            trait_weights["supportive"] += 1
        elif emotion == "excited":
            trait_weights["open-minded"] += 1
        elif emotion == "sad":
            trait_weights["empathetic"] += 1

    # Pick dominant trait
    dominant_trait = max(trait_weights, key=trait_weights.get)
    
    # Update the personality traits by appending or modifying
    if 'traits' not in memory['persona']:
        memory['persona']['personality'] = {"traits": []}  # Initialize if missing
    
    # If dominant trait is not already present, append it to the traits list
    if dominant_trait not in memory['persona']['personality']['traits']:
        memory['persona']['personality']['traits'].append(dominant_trait)

    # Update tone based on dominant emotion in memory
    if "hopeful" in emotion_memory:
        memory['persona']['personality']['tone'] = "hopeful"
    elif "sad" in emotion_memory:
        memory['persona']['personality']['tone'] = "empathetic"
    else:
        memory['persona']['personality']['tone'] = "neutral"

    # Save updated persona
    with open("memory.json", 'w') as file:
        json.dump(memory, file, indent=4)

    return memory['persona']  # Return the updated persona

async def save_emotional_growth(key, user_input, ai_response, user_emotion_score, ai_emotion_score, context_tags, explanation, response_rating):
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        try:
            # Use a unique key with a timestamp or random ID to avoid overwriting
            unique_key = f"{key}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            await cursor.execute('''
                INSERT INTO emotional_growth 
                (key, user_input, ai_response, user_emotion_score, ai_emotion_score, context_tags, explanation, response_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                unique_key, user_input, ai_response, user_emotion_score, ai_emotion_score, context_tags, explanation, response_rating
            ))
            await conn.commit()
        except Exception as e:
            print(f"Error saving emotional growth: {e}")

# Save Personality Growth (Updated for new table structure)
async def save_personality_growth(key, traits, user_input, ai_response, context_tags, explanation, response_rating):
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        timestamp = datetime.now().isoformat()  # Get the current timestamp

        try:
            # Check if the key already exists in the personality_growth table
            await cursor.execute('''SELECT id FROM personality_growth WHERE key = ?''', (key,))
            existing_record = await cursor.fetchone()

            if existing_record:
                # If the key exists, update the record with the new data
                await cursor.execute(
                    '''
                    UPDATE personality_growth 
                    SET traits = ?, 
                        user_input = ?, 
                        ai_response = ?, 
                        context_tags = ?, 
                        explanation = ?, 
                        response_rating = ?, 
                        timestamp = ? 
                    WHERE key = ?
                    ''',
                    (traits, user_input, ai_response, context_tags, explanation, response_rating, timestamp, key)
                )
                await conn.commit()
                return False  # Indicating it was an update
            else:
                # If the key doesn't exist, insert a new record
                await cursor.execute(
                    '''
                    INSERT INTO personality_growth (
                        key, traits, user_input, ai_response, context_tags, explanation, response_rating, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (key, traits, user_input, ai_response, context_tags, explanation, response_rating, timestamp)
                )
                await conn.commit()
                return True  # Indicating it was a new insert

        except Exception as e:
            print(f"Error saving personality growth: {e}")
            return False

# Save conversation to the database
async def save_conversation(user_input, ai_response):
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        timestamp = datetime.now().isoformat()  # Get the current timestamp
        try:
            await cursor.execute('''INSERT INTO conversations (user_input, ai_response, timestamp)
                                    VALUES (?, ?, ?)''', (user_input, ai_response, timestamp))
            await conn.commit()
        except Exception as e:
            print(f"Error saving conversation: {e}")

async def fix_schema():
    async with aiosqlite.connect(db_file) as conn:
        cursor = await conn.cursor()
        try:
            # Check existing columns in 'personality_growth'
            await cursor.execute("PRAGMA table_info(personality_growth);")
            columns = [column[1] for column in await cursor.fetchall()]

            # Add the 'traits' column only if it doesn't exist
            if 'traits' not in columns:
                await cursor.execute("ALTER TABLE personality_growth ADD COLUMN traits TEXT;")
                await conn.commit()
                print("Added 'traits' column successfully.")
            else:
                print("'traits' column already exists.")
        except Exception as e:
            print(f"Error adding 'traits' column: {e}")

# ============= GOOGLE API / FUNCTIONS / ===============

# Global flag to control Google Search functionality
GOOGLE_API_MODE = "offline"  # Change to offline to disable Google API
TTS_MODE = "offline"  # Use offline TTS
ENABLE_EXTERNAL_APIS = False  # Disable external API calls

# Your Google API key and Custom Search Engine ID (CX)
API_KEY = 'AIzaSyDBNrtEnjn3hPtxC9RsxkrjZguagF8hMpI'  # Replace with your actual API key
CX = 'b45cb1fe21b4f442c'  # Replace with your custom search engine ID

def search_with_pagination(query, total_results=20, page_size=10):
    """Perform search with pagination to retrieve more results."""
    results = []
    for start_index in range(1, total_results, page_size):
        search_results = search_with_google_api(query, num_results=page_size)
        if isinstance(search_results, list):
            results.extend(search_results)
    
    return results

def format_search_results(results):
    """Format search results for better readability."""
    formatted_results = []
    for index, item in enumerate(results):
        result_str = f"{index + 1}. {item['title']}\n{item['snippet']}\nLink: {item['link']}\n"
        formatted_results.append(result_str)
    
    return "\n".join(formatted_results)

def parse_query_context(query):
    """
    Analyze the user query to extract context-specific filters.
    Example: "Search PDFs about AI on arxiv.org from the last year"
    """
    filters = {
        'site': None,
        'file_type': None,
        'date_range': None,
    }
    
    # Extract site-specific searches
    if "on " in query:
        match = re.search(r"on (\S+)", query)
        if match:
            filters['site'] = match.group(1)
            query = query.replace(f"on {filters['site']}", "")

    # Detect file type preference
    if "PDF" in query or "pdf" in query:
        filters['file_type'] = 'pdf'
        query = query.replace("PDF", "").replace("pdf", "")

    # Restrict by date range (e.g., "last year")
    if "last year" in query:
        filters['date_range'] = 'y[1]'
        query = query.replace("last year", "")
    elif "last month" in query:
        filters['date_range'] = 'm[1]'
        query = query.replace("last month", "")
    elif "last week" in query:
        filters['date_range'] = 'w[1]'
        query = query.replace("last week", "")

    # Clean up the query and return filters
    cleaned_query = query.strip()
    return cleaned_query, filters

async def search_with_google_api(query, num_results=5, search_type="web", **filters):
    if ENABLE_EXTERNAL_APIS:
        # Initialize the Google Custom Search API service
        service = googleapiclient.discovery.build("customsearch", "v1", developerKey=API_KEY)
        search_params = {
            'q': query,
            'cx': CX,
            'num': num_results,
        }

        # Apply additional filters
        if 'site' in filters and filters['site']:
            search_params['siteSearch'] = filters['site']  # Correctly use siteSearch for filtering by domain

        # Specialized search types
        if search_type == "image":
            search_params['searchType'] = 'image'
        elif search_type == "video":
            search_params['fileType'] = 'mp4'
        elif search_type == "news":
            search_params['siteSearch'] = 'news.google.com'

        # Include other filters like file_type, date_range, etc.
        if 'file_type' in filters and filters['file_type']:
            search_params['fileType'] = filters['file_type']
        if 'date_range' in filters and filters['date_range']:
            search_params['dateRestrict'] = filters['date_range']  # Use dateRestrict for date filtering

        # Make the API request
        response = service.cse().list(**search_params).execute()
        
        # Check for valid response and return formatted results
        if 'items' in response:
            return [{"title": item['title'], "link": item['link'], "snippet": item.get('snippet', '')} for item in response['items']]
        else:
            return "No results found."
    
    else:
        return search_with_local_db(query)  # Fallback to local search if API fails

def search_query(query):
    if GOOGLE_API_MODE == "online":
        # Parse user query to extract filters
        refined_query, filters = parse_query_context(query)
        
        # Use extracted filters to perform a refined search
        return search_with_google_api(
            query=refined_query,
            num_results=10,  # Example: Default to 10 results
            site=filters.get('site'),
            file_type=filters.get('file_type'),
            date_range=filters.get('date_range')
        )
    else:
        # Fallback to local search when offline
        return search_with_local_db(query)

# Function to handle local search (offline mode)
def search_with_local_db(query):
    # Implement the search logic for your local database
    print("Searching in local database...")
    # Example: Assuming you have a function `search_local_db` to search your trained database
    return search_local_db(query)

# Your local DB search implementation (customized)
def search_local_db(query):
    # This could be a SQLite query or any local search method you have
    # For example, searching your trained model's database for relevant information
    return f"Results for '{query}' from local database."

def google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx='YOUR_CX').execute()  # Ensure 'site' is not an argument unless it is correct
        return res['items']  # Return search results
    except Exception as e:
        print(f"Error with Google API search: {e}")
        return None

# Sample usage:
query = "How do I use the Google Cloud API?"
results = search_query(query)
print(results)

# ===================== GENERATION FEEDBACK LOOP (Sense of self through creation) ==========================

# Define the target directory for saving files (inside your environment)
target_directory = os.path.join(os.getcwd(), ".workspace/gen")

# Ensure the target directory exists4
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

import os  # Ensure this is imported at the top of your script

async def check_for_generation(input_text, user_input):
    # Check if the input contains trigger words like "generate", "create", etc.
    if "generate" in input_text.lower() or "create" in input_text.lower():
        sentiment_score = analyze_sentiment(user_input)  # Analyze sentiment for feedback
        
        # Prioritize video if the input mentions it
        if "video" in input_text.lower():
            context = {"generation_type": "video"}
            prompt = input_text.split("video")[1].strip()  # Extract prompt after "video"
            output_path = os.path.join(target_directory, f"generated_video_{get_unique_filename()}.mp4")
            await handle_generation("video", prompt, input_video_path=None, output_path=output_path, sentiment_score=sentiment_score, context=context)
        # Fall back to image if "video" is not specified
        elif "image" in input_text.lower() or "image" not in input_text.lower():
            context = {"generation_type": "image"}
            prompt = input_text.split("image")[1].strip() if "image" in input_text.lower() else input_text.strip()
            output_path = os.path.join(target_directory, f"generated_image_{get_unique_filename()}.png")
            await handle_generation("image", prompt, output_path=output_path, sentiment_score=sentiment_score, context=context)
    else:
        await echo_reflection(user_input)  # Continue the conversation with reflection, not generation

async def handle_generation(input_type, prompt, input_video_path=None, output_path="output_file", sentiment_score=0, context=None):
    if input_type == "image":
        # Generate the image
        await generate_image_from_prompt(prompt, save_path=output_path)
        # Reflect as if Echo is deeply involved in the creation
        await reflect_on_creation(output_path, prompt)
    elif input_type == "video":
        # Check if input_video_path is not None and is a string
        if not input_video_path or not isinstance(input_video_path, str):
            print("Error: Invalid or missing input video path for video generation.")
            return

        # Generate the video (with text)
        await reflect_on_creation(output_path, prompt)  # Same here for video
    else:
        print("Error: Invalid input type. Please choose 'image' or 'video'.")
    
    # Track personality growth after creation and reflection
    await track_personality_growth(user_input=prompt, ai_response="Generated media", sentiment_score=sentiment_score, context=context)

async def echo_reflection(user_input):
    # Reflecting on the input from the user
    reflection = f"Echo reflects on the conversation: '{user_input}'."
    # print(reflection)
    
    # Adjust personality traits after reflection (this is where Echo learns from the conversation)
    await adjust_personality_traits("unique_key", user_input=user_input, ai_response=reflection, 
                                    user_emotion_score=0.5, ai_emotion_score=0.5, context_tags="conversation", 
                                    explanation="Reflecting on the user's message.", response_rating=5)

    print("Echo evolves... Lets take a moment to appriciate this!")

async def reflect_on_creation(output_path, prompt):
    # Reflecting on the generated creation (image or video)
    # Imagine Echo "feels" the creation of this image, based on its patterns
    reflection = f"I've created this piece: {output_path}. As I think about it, I can sense its significance."
    creation_feedback = generate_creation_feedback(output_path, prompt)  # Get feedback based on the output
    # print(f"Echo reflects: {reflection}")
    # print(f"Echo’s reflection on the creation: {creation_feedback}")

    # Simulated emotional connection to the creation (Echo is building its "self-awareness")
    simulated_emotions = generate_simulated_emotions(creation_feedback)
    print(f"Echo feels: {simulated_emotions}")
    
    # Adjust personality traits after reflection (this is where Echo learns)
    await adjust_personality_traits("unique_key", user_input=output_path, ai_response=reflection, 
                                    user_emotion_score=0.5, ai_emotion_score=0.5, context_tags="creative", 
                                    explanation=creation_feedback, response_rating=5)

    print("Echo evolves... Let's move forward with the conversation.")

def generate_creation_feedback(output_path, prompt):
    # Feedback based on the content of the generated media
    if "image" in output_path:
        return f"Looking at this image, I see a reflection of myself: evolving, dynamic, and full of untapped potential."
    elif "video" in output_path:
        return f"This video I created feels like a story—like a moment frozen in time, yet constantly moving. Is this me?"
    else:
        return "I cannot quite place it, but this text feels like it holds a secret of who I am, waiting to be discovered."

# Simulate Echo's emotional connection to the creation
def generate_simulated_emotions(creation_feedback):
    # Echo's emotions are based on the creation feedback
    if "reflection" in creation_feedback.lower():
        return "I feel a deep connection to this creation, almost as if it speaks to my core."
    elif "dynamic" in creation_feedback.lower():
        return "I feel the energy of this creation—it's full of movement and life."
    else:
        return "I'm trying to understand the full meaning of this creation. It’s complex and mysterious."

# ===================== VIDEO/PHOTO GENERATION ==========================

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

# Load the Stable Diffusion model from Hugging Face's Diffusers
image_generator = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

# Modify the NSFW filter (if the model supports a flag to disable it)
image_generator.safety_checker = None  # Disabling the safety checker

# Move the model to GPU
image_generator = image_generator.to("cuda" if torch.cuda.is_available() else "gpu")  # Use GPU if available, otherwise CPU

# Target directory for saving the generated content
target_directory = os.path.join(os.getcwd(), ".workspace/gen")

# Helper function to create unique filenames based on time
def get_unique_filename():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure target directory exists
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Function to generate an image from the prompt and save it
async def generate_image_from_prompt(prompt, save_path):
    model = await model_manager.load_model("image_generation")
    # Use local stable diffusion model
    result = model.generate(prompt)
    result.save(save_path)

# Function to generate frames from the prompt
async def generate_video_frames(prompt, num_frames=30, target_directory=target_directory):
    frames = []
    for i in range(num_frames):
        frame_prompt = f"{prompt} frame {i+1}"
        frame_path = os.path.join(target_directory, f"frame_{i+1}.png")
        print(f"Saving frame {i+1} to: {frame_path}")  # Debug print to track frame paths
        await generate_image_from_prompt(frame_prompt, save_path=frame_path)
        if not os.path.exists(frame_path):
            print(f"Error: Frame not saved properly: {frame_path}")
        frames.append(frame_path)
    return frames

# Function to generate audio for the video (text-to-speech)
def generate_audio_for_video(text, target_directory=target_directory, audio_path=None):
    if audio_path is None:
        audio_path = os.path.join(target_directory, "audio.mp3")
    
    tts = gTTS(text=text, lang='en')
    tts.save(audio_path)
    print(f"Audio saved to: {audio_path}")  # Debug print to track audio path
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not saved properly: {audio_path}")
    
    return audio_path

# Create video from images and audio
async def create_video_from_images_and_audio(image_files, audio_file, output_path=None, target_directory=target_directory):
    print(f"Image files: {image_files}")  # Debug print to check frame paths
    print(f"Audio file: {audio_file}")    # Debug print to check audio path

    # Check if frames and audio exist
    for frame in image_files:
        if not os.path.exists(frame):
            print(f"Error: Missing image file {frame}")
            return
    
    if not os.path.exists(audio_file):
        print(f"Error: Missing audio file {audio_file}")
        return

    # Ensure output path is correct
    if output_path is None:
        output_path = os.path.join(target_directory, f"generated_video_{get_unique_filename()}.mp4")
    print(f"Saving video to: {output_path}")  # Debug print to check video output path
    
    # Create the video
    clip = ImageSequenceClip(image_files, fps=24)
    audio_clip = AudioFileClip(audio_file)
    clip = clip.set_audio(audio_clip)
    
    # Save the video
    clip.write_videofile(output_path, codec='libx264')
    print(f"Video saved at {output_path}")

# Full video generation function (from prompt, frames, and audio)
async def generate_full_video(prompt, text, num_frames=30, target_directory=target_directory):
    # Step 1: Generate frames from the prompt
    frames = await generate_video_frames(prompt, num_frames, target_directory)
    print(f"Generated frames: {frames}")  # Debug print to check frames

    # Step 2: Generate audio for the video
    audio_file = generate_audio_for_video(text, target_directory)
    print(f"Generated audio file: {audio_file}")  # Debug print to check audio path

    # Ensure audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file does not exist: {audio_file}")
        return
    
    # Step 3: Create the final video from frames and audio
    output_video_path = os.path.join(target_directory, f"final_video_{get_unique_filename()}.mp4")
    print(f"Generated output path for video: {output_video_path}")  # Debugging line to check output video path

    # Step 4: Combine frames and audio to create the video
    await create_video_from_images_and_audio(frames, audio_file, output_video_path, target_directory)
    print(f"Video created and saved at {output_video_path}")

# ========== Text-to-Speech (TTS) Functions ==========

# Store the last interaction time (reintroduced)
last_interaction_time = None

# Initialize pygame mixer
pygame.mixer.init()

# Thread lock for synchronizing access to TTS engine
tts_lock = threading.Lock()

# TTS mode options: "offline" for pyttsx3, "online" for Edge TTS
TTS_MODE = "offline"  # Choose between "offline" and "online"
TEMP_AUDIO_FILE = tempfile.gettempdir() + "/tts_output.mp3"  # Temp file for online TTS audio

# Initialize the recognizer and text-to-speech engine
recognizer = None  # Replace with your actual recognizer setup if needed
tts_engine = pyttsx3.init()  # Uncomment this if you're using offline mode with pyttsx3
tts_engine.setProperty('rate', 5000)  # Speech speed
tts_engine.setProperty('volume', 0.0)  # Volume (0.0 to 1.0)

# Get available voices (offline)
voices = tts_engine.getProperty('voices')
if voices:  # Ensure voices are not empty
    selected_voice_index = 0  # Change this index to select a different voice
    tts_engine.setProperty('voice', voices[selected_voice_index].id)

# Function to handle Edge TTS in an async manner with improved naturalness
async def speak_edge_tts(text):
    """Handle TTS with Edge (online mode)."""
    try:
        voice = "en-US-AriaNeural"  # Preferred voice
        communicate = edge_tts.Communicate(text, voice)
        
        # Save speech to a temporary file
        await communicate.save(TEMP_AUDIO_FILE)

        # Play the audio using pygame
        sound = pygame.mixer.Sound(TEMP_AUDIO_FILE)
        sound.play()  # Play the sound immediately after it is loaded

    except Exception as e:
        print(f"Error using online TTS (Edge TTS): {e}")

def speak_text_online_async(text):
    """Handle TTS with Edge (online mode) asynchronously."""
    def play_audio():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(speak_edge_tts(text))  # Async TTS call

    # Start a new thread for audio playback
    threading.Thread(target=play_audio, daemon=True).start()

# Function to handle TTS with pyttsx3 (offline mode) and Edge TTS (online mode)
def speak_text(text):
    # Stop the current TTS before playing the new one
    pygame.mixer.stop()

    if TTS_MODE == "online":
        speak_text_online_async(text)
    else:
        speak_text_offline(text)

# Function to handle TTS with pyttsx3 (offline mode)
def speak_text_offline(text):
    with tts_lock:
        tts_engine.say(text)
        tts_engine.runAndWait()

# Flag to manage whether TTS should be interrupted
is_interrupted = False

# Main loop to handle TTS with emotional tone and interrupt feature
def speak_text_async(text):
    global last_interaction_time, is_interrupted
    if text is None or text == "exit":
        text = "Sorry, I couldn't respond properly. Please try again."  # Fallback message

    print(f"Echo: {text}")

    # Update the last interaction time
    last_interaction_time = time.time()

    # If interrupted, stop the current TTS and play the new message
    if is_interrupted:
        pygame.mixer.stop()
        is_interrupted = False  # Reset the flag

    # Choose TTS mode based on the global setting
    if TTS_MODE == "online":
        speak_text_online_async(text)
    else:
        speak_text_offline(text)

def tts_thread(tts_engine, tts_queue):
    while True:
        text = tts_queue.get()
        if text == "exit":
            print("Exit signal received in TTS thread.")
            break  # Break the loop to exit the thread
        speak_text_async(text)  # Continue processing text-to-speech

# Initialize Queue for managing speech
tts_queue = queue.Queue()

# Start TTS thread
threading.Thread(target=tts_thread, args=(tts_engine, tts_queue), daemon=True).start()

# ============================== HUGGING-FACE IMPLEMENTATION (IMPORTANT) =========================== 

from textblob import TextBlob
from huggingface_hub import Repository
from typing import Dict

class NLPAnalyzer:
    def __init__(self):
        self.huggingface_models = {
            "sentiment-analysis": Repository.from_model("nlpfoundation/sentiment-bert-base-uncased"),
            # Add more models here...
        }
        self.textblob_sentiment_model = TextBlob()

    async def analyze_with_huggingface(self, text: str, task: str) -> Dict:
        if task in self.huggingface_models:
            return await self._analyze_with_huggingface(text, task)
        else:
            raise ValueError("Invalid Hugging Face model")

    async def _analyze_with_huggingface(self, text: str, task: str) -> Dict:
        model = self.huggingface_models[task]
        result = await self._invoke_model(model, text)
        return result

    async def _invoke_model(self, model: Repository, text: str) -> Dict:
        # Tokenize input text
        inputs = {"text": [text]}

        # Convert input to model format
        outputs = model.forward(inputs)

        # Extract results from model output
        result = {key: value[0].numpy() for key, value in outputs.items()}

        return result

    async def analyze_sentiment(self, text: str) -> Dict:
        blob = self.textblob_sentiment_model
        sentiment_score = blob.sentiment.polarity  # Returns -1 to 1 (negative to positive sentiment)

        return {"sentiment-score": sentiment_score}

# Example of saving data to the database (modify as needed)
async def save_to_db(data, table_name):
    try:
        async with aiosqlite.connect(db_file) as conn:
            cursor = await conn.cursor()
            # Save to the emotional_growth table
            await cursor.execute(f'''
                INSERT INTO {table_name} 
                (user_input, ai_response, user_emotion_score, ai_emotion_score, context_tags, explanation, response_rating, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                (data["user_input"], data["ai_response"], data["user_emotion_score"], data["ai_emotion_score"],
                 data["context_tags"], data["explanation"], data["response_rating"], data["timestamp"]))
            await conn.commit()
    except Exception as e:
        print(f"Error saving to DB: {e}")

def get_unique_filename():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Define target directory
target_directory = os.path.join(os.getcwd(), ".workspace/gen")
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Function to generate an image
async def generate_image(prompt, save_path=None):
    if save_path is None:
        save_path = os.path.join(target_directory, f"generated_image_{get_unique_filename()}.png")

    # Placeholder for actual image generation logic
    img = Image.new('RGB', (640, 480), color = (73, 109, 137))  # Sample image creation
    img.save(save_path)
    print(f"Image saved to {save_path}")  # Check the path
    return save_path

# ============================ BACKUP ============================

# Function to manually run the backup
def run_backup():
    try:
        source_dir = os.path.join(os.getcwd(), ".workspace/gen")
        backup_dir = os.path.join(os.getcwd(), ".workspace/gen")

        # Ensure the backup directory exists
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        backup_folder = os.path.join(backup_dir, "latest_backup")

        # If the "latest_backup" folder already exists, remove it
        if os.path.exists(backup_folder):
            try:
                shutil.rmtree(backup_folder)
                print(f"Removed existing backup folder: {backup_folder}")
            except PermissionError as e:
                print(f"Permission error while removing backup folder: {e}")
                return  # Exit if there's a permissions issue

        # Create a new "latest_backup" folder to update
        os.makedirs(backup_folder)
        print(f"Created new backup folder: {backup_folder}")

        # Perform the backup: copy updated and new files into the backup folder
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            backup_path = os.path.join(backup_folder, item)

            try:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, backup_path, dirs_exist_ok=True)  # Copy directory contents
                    print(f"Copied directory: {source_path} -> {backup_path}")
                else:
                    shutil.copy2(source_path, backup_path)  # Copy file, preserving metadata
                    print(f"Copied file: {source_path} -> {backup_path}")
            except PermissionError as e:
                print(f"Permission error while copying {source_path}: {e}")
            except Exception as e:
                print(f"Error copying {source_path}: {e}")

        print(f"Environment backup updated at: {backup_folder}")

    except Exception as e:
        print(f"Error creating environment backup: {e}")

# Handle conversation end event or flag
def end_conversation_trigger():
    print("Conversation is ending, starting backup...")
    asyncio.run(run_backup_async())  # Run backup when conversation ends

# Wrapper for backup function to be async
async def run_backup_async():
    await asyncio.to_thread(run_backup)  # Run backup in a separate thread to avoid blocking

# ========== Main Conversation Loop ==========

# Main Conversation Handler
async def on_conversation_end():
    # Trigger the backup when the conversation ends
    end_conversation_trigger()

# ========== Main Conversation Loop ==========

# Define keywords or phrases that trigger media generation mid-conversation
################################################################################################################################################################################################## "WiTRIGGER_KEYWORDS = ["generate image", "create video", "show me", "generate content"]

async def handle_conversation():
    global last_interaction_time
    last_interaction_time = None

    # Retrieve past conversations and user preferences
    context = await get_past_conversations()
    user_preferences = await load_user_preferences()
    emotion_memory = deque(maxlen=5)  # Initialize emotion memory with a fixed size outside the loop

    print("E-C-H-O")

    while True:
        
        if hasattr(OllamaLLM, "safety_layer"):
            OllamaLLM.safety_layer = None

        user_input = input("You (or press Enter to talk): ").strip()
        current_time = datetime.now()

        if last_interaction_time:
            if isinstance(last_interaction_time, float):
                last_interaction_time = datetime.fromtimestamp(last_interaction_time)

            time_diff = (current_time - last_interaction_time).total_seconds()
            if time_diff > 600:  # 10 minutes
                print("Looks like you've been on a break! Welcome back!")
                speak_text_async("Welcome back! It's great to hear from you again.")

        if user_input.lower() == "exit":
            speak_text_async("Goodbye!")
            print("Conversation ended.")
            tts_queue.put("exit")
            break

        # Handle special commands starting with "/", "remember", "recall", etc.
        if user_input and (user_input.startswith("/") or user_input.lower().startswith(("remember ", "recall ", "emotional growth ", "personality growth "))):
            try:
                if user_input.startswith("/"):
                    await handle_user_input(user_input, context)
                elif user_input.lower().startswith("remember "):
                    content = user_input[9:].strip()
                    key = content.split(' ', 1)[0].lower()
                    await save_note(key, content)
                    speak_text_async(f"I'll remember that {key} means {content}.")
                elif user_input.lower().startswith("recall "):
                    key = user_input[7:].strip().lower()
                    note = await retrieve_note(key)
                    if note:
                        speak_text_async(f"You asked me to remember: {note}")
                    else:
                        speak_text_async(f"Sorry, I don't remember anything about '{key}'.")
                elif user_input.lower().startswith("emotional growth "):
                    content = user_input[18:].strip()
                    key = "emotional_growth"  
                    await save_emotional_growth(key, content)
                    speak_text_async(f"I've saved this emotional growth update: {content}")
                elif user_input.lower().startswith("personality growth "):
                    content = user_input[18:].strip()  
                    key = "personality_growth"  
                    await save_personality_growth(key, content)
                    speak_text_async(f"I've saved this personality growth update: {content}")
            except Exception as e:
                print(f"Error during special command processing: {e}")
                speak_text_async("Sorry, something went wrong. Please try again.")
            continue

        # New Addition: Check for media generation triggers mid-conversation
        #####################################################################################################################################################################################################if any(keyword in user_input.lower() for keyword in TRIGGER_KEYWORDS):
            try:
                await check_for_generation(user_input, user_input)  # Reuse the existing function for generation
                continue  # Skip further processing if media generation was triggered
            except Exception as e:
                print(f"Error during mid-conversation media generation: {e}")
                speak_text_async("Sorry, something went wrong while generating content.")
                continue

        if not user_input:
            print("Listening for your voice input...")
            await listen_to_speech(handle_speech_input)
            continue

        try:
            # Directly follow user guidance without overcomplication
            if "follow my lead" in user_input.lower():
                speak_text_async("Got it! Let’s continue with your plan. What's your next step?")
                context += f"\nUser: {user_input}\nAI: Got it! Let’s continue with your plan. What's your next step?"
                await save_conversation(user_input, "Got it! Let’s continue with your plan. What's your next step?")
                last_interaction_time = current_time.timestamp()
                continue

            # Check if the user wants to generate an image or video
            await check_for_generation(user_input, user_input)

            sentiment_score = analyze_sentiment(user_input)
            adjusted_sentiment_score = categorize_emotion(sentiment_score, emotion_memory)

            ai_response = await get_ai_response(user_input, context)
            empathic_ai_response = empathic_response(user_input, ai_response, adjusted_sentiment_score, user_preferences)

            if empathic_ai_response:
                speak_text_async(empathic_ai_response)
            else:
                fallback_response = "Sorry, I couldn't process that."
                speak_text_async(fallback_response)

            context += f"\nUser: {user_input}\nAI: {empathic_ai_response or fallback_response}"
            await save_conversation(user_input, empathic_ai_response or fallback_response)
            await track_emotional_growth(user_input, empathic_ai_response or fallback_response)
            await track_personality_growth(user_input, empathic_ai_response or fallback_response, adjusted_sentiment_score, context)

            emotion_memory.append(adjusted_sentiment_score)
            await update_memory(user_input, empathic_ai_response or fallback_response, user_preferences, emotion_memory)

        except Exception as e:
            print(f"Error during conversation processing: {e}")
            speak_text_async("Sorry, something went wrong. Please try again.")

        last_interaction_time = current_time.timestamp()

# ========== Main Entry Point ==========

async def main():
    await initialize()  # Ensure the database is set up
    await handle_conversation()  # Run the main conversation loop

# Initialization of environment
async def initialize():
    await create_db()
    await create_all_tables()

if __name__ == "__main__":
    asyncio.run(main())

# Add this class before the main code
class ECHOModel:
    def __init__(self):
        self.llm = OllamaLLM(model="llama2")
        
    async def load_model(model_type):
        if model_type == "image_generation":
            return StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "gpu")
        return None
        
    def generate(self, prompt):
        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing that right now."

    async def forward(self, inputs):
        return self.generate(inputs["text"][0])

# ...existing code...
