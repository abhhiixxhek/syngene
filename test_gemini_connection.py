import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print(f"✅ API Key found: {api_key[:5]}...")

models_to_test = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash-exp"]

print("\n--- Testing Model Availability ---")
for model_name in models_to_test:
    print(f"\nTrying model: {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, exists?")
        print(f"✅ Success! {model_name} is working.")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Failed: {model_name}")
        print(f"Error: {e}")

print("\n--- Listing Available Models (if possible) ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
