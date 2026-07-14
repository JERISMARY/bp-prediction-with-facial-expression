# Blood Pressure Prediction Chatbot

A Flask-based web application for blood pressure prediction with AI-powered chatbot assistance using Google's Gemini API.

## Features

- Blood pressure prediction using machine learning
- AI chatbot for hypertension education and guidance
- Voice and face analysis integration
- Firebase deployment ready

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

The Gemini API is already configured with a valid API key. The chatbot will work with both rule-based responses (fast) and AI-powered responses (intelligent) when quota is available.

## Gemini API Setup

The chatbot uses Google's Gemini 2.0 Flash model for intelligent responses.

**API Key Status:** ✅ Configured with `AIzaSyD5cu4By4hOeQMyUkOlWP9zAewBMmo7vY0`

**Note:** The current API key may have quota limitations. If you encounter quota exceeded errors, you can:
- Wait for quota reset (usually daily)
- Upgrade to a paid Google AI plan
- Generate a new API key with fresh quota

The system will automatically fall back to rule-based responses if Gemini is unavailable.

## Chatbot Integration

The chatbot uses a **smart three-tier system**:

1. **Rule-Based Responses** (instant):
   - Fast, reliable answers for common hypertension topics
   - Always available, no API calls needed

2. **Gemini AI Responses** (intelligent):
   - Advanced AI responses for complex queries
   - Available when API quota allows

3. **Graceful Fallbacks** (helpful):
   - Clear guidance when AI is unavailable
   - Explains what topics the chatbot can help with
   - Encourages consulting healthcare providers

**Current Status:** Rule-based responses are fully functional. AI responses available when quota resets.
- **System prompt:** Specialized for hypertension education with empathetic, concise responses

## API Endpoints

- `GET /` - Main application page
- `POST /predict` - Blood pressure prediction
- `POST /chat` - Chatbot interaction
- `POST /analyze-image` - Lab report image analysis

## Deployment

The application is configured for deployment on:
- Google Cloud Run
- Firebase Functions
- Heroku

## Security Note

The default API key in the code has been reported as leaked. Please obtain your own API key from Google AI Studio and set it as an environment variable."# BP-FACIAL-PREDICTION" 
