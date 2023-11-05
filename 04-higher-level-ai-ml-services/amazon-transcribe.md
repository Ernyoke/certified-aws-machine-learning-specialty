# Amazon Transcribe

- It transcribes speech to text
- Speech input can be in FLAC, MP3, MP4 or WAV formats in a specified language
- Supports streaming audio (HTTP/2, WebSockets) for French, English and Spanish
- Can identify and distinct speakers
- Can do channel identification:
    - Example: two callers could be transcribed separately
- Can automatically detect language (it can detect the dominant one spoken)
- Supports custom vocabulary
    - Vocabulary Lists: just a list a special words
    - Vocabulary Tables: can include `SoundsLike`, `IPA` and `DisplayAs`

## Use Cases

- Call Analytics:
    - Trained specifically for customer service and sales calls
    - Real-time transcriptions and insights
    - Sentiment, talk speed, interruptions, look for specific phrases
- Medical:
    - Trained on medical terminology
    - HIPAA-eligible
- Subtitling:
    - Live subtitle output