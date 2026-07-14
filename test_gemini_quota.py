import requests
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(">>> .env file loaded")
except ImportError:
    print(">>> python-dotenv not installed")

# Check Gemini API quota status
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyD5cu4By4hOeQMyUkOlWP9zAewBMmo7vY0')

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Simple test request
payload = {
    "contents": [{
        "parts": [{"text": "Hello"}]
    }],
    "generationConfig": {
        "temperature": 0.2,
        "maxOutputTokens": 50
    }
}

try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    response_text = response.text
    print(f"Response: {response_text[:1000]}...")

    if response.status_code == 429:
        print("\n❌ QUOTA EXCEEDED - API limit reached")
        print("💡 Solutions:")
        print("   1. Wait for quota reset (usually daily)")
        print("   2. Get a new API key from Google AI Studio")
        print("   3. Upgrade to a paid plan")
    elif response.status_code == 400:
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            print(f"\n❌ API KEY ERROR: {error_message}")
            if "expired" in error_message.lower() or "renew" in error_message.lower():
                print("💡 Solution: Get a new API key from https://aistudio.google.com/app/apikey")
            else:
                print("💡 Solution: Check your API key and try again")
                print("💡 Solution: Check your API key and try again")
        except:
            print("\n❌ API KEY ERROR: Unable to parse error response")
            print("💡 Solution: Get a new API key from https://aistudio.google.com/app/apikey")
    elif response.status_code == 200:
        print("\n✅ API is working - quota available")
    else:
        print(f"\n⚠️ Unexpected status: {response.status_code}")

except Exception as e:
    print(f"❌ Error testing API: {e}")