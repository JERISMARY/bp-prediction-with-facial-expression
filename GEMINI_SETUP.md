# Gemini API Setup Guide

## Current Issue: Quota Exceeded
Your chatbot is working correctly, but the Google Gemini API quota has been exceeded. This is why you're seeing fallback messages instead of AI responses.

## Solutions

### Option 1: Get a New API Key (Recommended)
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the new API key
5. Set it as an environment variable:
   ```bash
   export GEMINI_API_KEY=your_new_api_key_here
   ```
6. Or create a `.env` file in your project root:
   ```
   GEMINI_API_KEY=your_new_api_key_here
   ```

### Option 2: Wait for Quota Reset
- Free tier quotas typically reset daily
- Check your usage at: https://ai.google.dev/rate-limits

### Option 3: Upgrade to Paid Plan
- Visit [Google AI Studio Billing](https://ai.google.dev/pricing)
- Higher limits and faster processing

## Testing Your Setup
After setting up a new API key, test it with:
```bash
python test_gemini_quota.py
```

You should see "✅ API is working - quota available" instead of the quota exceeded error.

## Current Status
- ✅ Rule-based responses: Working perfectly
- ✅ Chat interface: Fully functional
- ✅ Facial analysis: Working (may also be affected by quota)
- ❌ AI chat responses: Blocked by quota
- ❌ Meal rating: Blocked by quota

The chatbot provides excellent rule-based responses for common hypertension questions, so users still get helpful information!