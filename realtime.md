Real-Time Emotion Analysis Implementation Guide
📋 Executive Summary
Goal: Transform the current batch-processing emotion analysis system into a real-time streaming application where users can speak into their microphone and see live sentiment/emotion analysis from both text (Gladia) and voice (local ML model).
Current State: Offline analysis of pre-recorded audio files
Target State: Live microphone input with real-time dual-source emotion detection
🔍 Current Codebase Architecture
What We Have Now
The existing codebase is structured as separate command-line tools that process audio files in a pipeline:
1. download_translation.py   └─> Downloads completed Gladia transcription jobs (REST API)   └─> Saves JSON with text + sentiment/emotion analysis2. emotion_recognition.py     └─> Loads pre-recorded audio file (e.g., Rocky.mp3)   └─> Loads pre-existing Gladia JSON segments   └─> Runs local WavLM model on each audio segment   └─> Outputs comparison JSON (mapped_emotions.json)3. plot_sentiment.py   └─> Reads mapped_emotions.json   └─> Generates static PNG visualizations4. index.html + visuals.js   └─> Static web viewer   └─> Loads hardcoded audio file (Rocky.mp3)   └─> Loads hardcoded JSON (mapped_emotions.json)   └─> Shows waveform with pre-computed results
Key Characteristics:
All processing happens before the user opens the web page
Uses Gladia's batch transcription API (submit → poll → download)
Model inference runs once per segment from saved audio
Frontend is read-only (no user interaction beyond playback)
🎯 Target Architecture
What We're Building
A real-time WebSocket-based application with three components:
┌─────────────────────────────────────────────────┐│  FRONTEND (Browser)                             ││  ┌─────────────────────────────────────────┐   ││  │ • Captures microphone audio             │   ││  │ • Sends chunks via WebSocket            │   ││  │ • Displays results as they stream in    │   ││  └─────────────────────────────────────────┘   │└────────────────┬────────────────────────────────┘                 │ WebSocket                 │ (Audio chunks + JSON responses)                 ▼┌─────────────────────────────────────────────────┐│  BACKEND SERVER (Python FastAPI)                ││  ┌─────────────────────────────────────────┐   ││  │ [Startup: Load ML model once]           │   ││  │                                          │   ││  │ [Per Connection:]                        │   ││  │  • Receive audio chunk every ~2 seconds │   ││  │  • Fork into two parallel paths:        │   ││  │                                          │   ││  │    Path A: Forward to Gladia WS ────┐   │   ││  │                                      │   │   ││  │    Path B: Run local inference ─────┤   │   ││  │                                      │   │   ││  │  • Stream results back to frontend   │   │   ││  └──────────────────────────────────────┘   │└──────────┬──────────────────┬─────────────────┘           │                  │           ▼                  ▼    ┌──────────────┐   ┌─────────────────┐    │ Gladia       │   │ Local WavLM     │    │ Real-Time    │   │ Model           │    │ WebSocket    │   │ (CPU inference) │    │ API          │   │                 │    └──────────────┘   └─────────────────┘
🔄 Required Changes by File
Files to CREATE (New)
1. realtime_server.py (NEW - ~300 lines)
Purpose: WebSocket server that orchestrates real-time processing
Key Responsibilities:
Startup phase: Load the SER model into memory once (happens before any requests)
Per-connection phase:
Accept WebSocket connection from frontend
Establish WebSocket connection to Gladia's real-time API
Maintain audio buffer for chunking
Process incoming audio in two parallel streams:
Stream A: Forward raw audio to Gladia → receive transcription + text-based emotion
Stream B: Buffer audio into 2-second chunks → run local model → get voice-based emotion
Send both result types back to frontend as they arrive (not synchronized)
Key Functions:
load_model(): FastAPI startup event, loads model globally
process_audio_chunk(): Runs inference on 2-second PCM audio buffer
websocket_analyze(): Main WebSocket endpoint handling bidirectional communication
gladia_realtime_handler(): Async task forwarding Gladia responses to frontend
Integration Points:
Imports from vad2gladia.py → map_vad_to_gladia() function
Imports from download_translation.py → load_api_key() function
Uses existing model ID: "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"
2. realtime.html (NEW - ~200 lines)
Purpose: Browser interface for live recording and display
Key Responsibilities:
Request microphone permission via navigator.mediaDevices.getUserMedia()
Setup AudioContext to capture raw PCM audio at 16kHz
Use ScriptProcessorNode to process audio chunks (8192 samples at a time)
Convert Float32Array → Int16Array → Base64 → send via WebSocket
Receive JSON responses from server and update UI in real-time
Display two side-by-side cards:
Gladia card: Shows transcribed text + text-based sentiment/emotion
Voice card: Shows VAD values + voice-based sentiment/emotion
Visual mismatch detection: Highlight cards when sentiment/emotion disagree
WebSocket Message Protocol:
// Client → Server{  "type": "audio",  "data": "<base64-encoded-pcm>",  "sample_rate": 16000}// Server → Client (Gladia results){  "source": "gladia",  "text": "transcribed text...",  "sentiment": "negative",  "emotion": "anger",  "is_final": true}// Server → Client (Voice results){  "source": "voice",  "sentiment": "neutral",  "emotion": "contempt",  "confidence": 0.894,  "vad": {    "valence": -0.367,    "arousal": 0.204,    "dominance": 0.254  }}
Files to MODIFY (Existing)
3. emotion_recognition.py (MODIFY - Extract logic)
Changes needed:
Extract the model loading logic into reusable functions
Extract the segment processing logic to work on raw numpy arrays (not just file paths)
Make the inference pipeline chunk-friendly (accept audio bytes, not just file paths)
Specific modifications:
# Add new function (keep existing code)def load_ser_model(device: torch.device) -> tuple:    """    Extracted model loading logic for reuse in server.    Returns (model, config_dict)    """    model = AutoModelForAudioClassification.from_pretrained(        MODEL_ID,        trust_remote_code=True,    ).to(device)    model.eval()        config = {        "sampling_rate": int(getattr(model.config, "sampling_rate", 16_000)),        "mean": float(getattr(model.config, "mean", 0.0)),        "std": float(getattr(model.config, "std", 1.0)),    }        return model, config# Add new function for chunk processingdef process_audio_array(    audio_array: np.ndarray,    model: torch.nn.Module,    config: dict,    device: torch.device) -> dict:    """    Run inference on a numpy audio array.    Used by real-time server for chunk processing.    """    # Existing normalization + inference logic from run_inference()    # but accepts numpy array instead of loading from file    pass
Why: The real-time server needs to reuse the same model loading and inference logic without duplicating code.
4. download_translation.py (MODIFY - Make key loading reusable)
Changes needed:
The load_api_key() function is already good
Ensure it's importable and doesn't have side effects when imported
Verification:
# Should work without running main()from download_translation import load_api_keyapi_key = load_api_key()  # Returns string, no execution
Why: The real-time server needs the Gladia API key to establish WebSocket connection.
5. vad2gladia.py (NO CHANGES NEEDED)
Status: Already perfect for reuse
Why: The map_vad_to_gladia() function is already a pure function that accepts VAD values and returns mapped emotions. The real-time server will call it directly.
Files to KEEP AS-IS (Unchanged)
plot_sentiment.py - Still useful for generating reports from saved sessions
index.html + visuals.js - Keep for reviewing pre-recorded analyses
mapped_emotions.json - Example data for documentation
🔌 Technical Integration Details
Audio Format Requirements
What the browser captures:
Format: Raw PCM (Pulse Code Modulation)
Encoding: 16-bit signed integers
Channels: Mono (1 channel)
Sample rate: 16,000 Hz
Byte order: Little-endian
What Gladia real-time API expects:
Same format as above (PCM 16kHz mono)
Sent as raw bytes over WebSocket
Configuration sent in first message
What the local model expects:
Numpy array of float32 values in range [-1.0, 1.0]
Sample rate: 16,000 Hz (resampled if different)
Normalized using model's mean/std values
Conversion pipeline in server:
Raw bytes from browser (Int16)  → Convert to numpy array  → Normalize to [-1.0, 1.0] (divide by 32768)  → Apply model normalization (mean/std)  → Convert to torch tensor  → Run inference
Gladia Real-Time API Integration
Endpoint: wss://api.gladia.io/audio/text/audio-transcription
Connection Flow:
Connect to WebSocket with x-gladia-key header
Send configuration JSON as first message:
   {     "encoding": "WAV/PCM",     "sample_rate": 16000,     "language_behaviour": "automatic single language",     "frames_format": "bytes",     "sentiment_analysis": true,     "enable_emotions": true   }
Stream audio bytes continuously
Receive transcription results asynchronously
Response Format:
{  "type": "transcript",  "transcription": "hello world",  "is_final": false,  "confidence": 0.95,  "sentiment": "neutral",  "emotion": "neutral"}
Documentation: Check Gladia's real-time API docs for latest format
Performance Considerations
Model Loading (One-time cost):
Time: 30-60 seconds on first startup
Memory: ~500MB RAM + ~1.5GB for model weights
CPU: No GPU needed, runs on CPU
Per-chunk Inference (Every 2 seconds):
Input: 2 seconds of audio = 32,000 samples = 64KB
Processing time on CPU: ~200-500ms per chunk
This is acceptable since chunks arrive every 2000ms
Buffering Strategy:
chunk_duration = 2.0  # secondssample_rate = 16000   # Hzbytes_per_sample = 2  # 16-bit = 2 bytesbytes_per_chunk = int(chunk_duration * sample_rate * bytes_per_sample)# = 2.0 * 16000 * 2 = 64,000 bytes per chunk# Accumulate audio until we have enoughaudio_buffer = bytearray()while receiving:    audio_buffer.extend(new_audio_bytes)    if len(audio_buffer) >= bytes_per_chunk:        process_chunk(audio_buffer[:bytes_per_chunk])        audio_buffer = audio_buffer[bytes_per_chunk:]
🚀 Implementation Steps
Phase 1: Backend Server Setup
Create realtime_server.py with basic FastAPI structure
Implement model loading in @app.on_event("startup")
Create WebSocket endpoint /ws/analyze
Test model loading works and server starts
Phase 2: Audio Processing Pipeline
Extract reusable functions from emotion_recognition.py
Implement process_audio_chunk() function
Test with sample audio bytes (use existing audio file for testing)
Verify VAD values and emotion mapping work correctly
Phase 3: Gladia Integration
Implement Gladia WebSocket connection in server
Test forwarding audio to Gladia and receiving responses
Parse Gladia responses and extract sentiment/emotion
Handle connection errors and reconnection logic
Phase 4: Frontend Development
Create realtime.html with microphone capture
Implement audio chunking using AudioContext
Setup WebSocket client communication
Create UI cards for displaying results
Phase 5: Integration & Testing
Test end-to-end: speak → see results
Verify both sources update independently
Implement mismatch detection visualization
Test error handling (network failures, permission denials)
Phase 6: Polish
Add loading states and status indicators
Improve error messages for users
Test on different browsers (Chrome, Firefox, Safari)
Document API key setup and deployment
🔧 Development Environment Setup
# Install new dependenciespip install fastapi uvicorn[standard] websockets# Start server (terminal 1)python realtime_server.py# Wait for "Model loaded" message# Serve frontend (terminal 2)python -m http.server 3000# Open browser# Navigate to http://localhost:3000/realtime.html
🎯 Success Criteria
The implementation is complete when:
✅ User can click "Start Recording" and grant microphone access
✅ Audio streams from browser to server continuously
✅ Gladia transcription appears in real-time as user speaks
✅ Voice-based emotion updates every ~2 seconds
✅ Both cards show independent results (timestamps may differ)
✅ Visual mismatch indicator activates when sentiment/emotion disagree
✅ Server handles multiple concurrent connections
✅ Model only loads once at startup (not per connection)
⚠️ Known Limitations & Trade-offs
Timestamp misalignment: Gladia and local results won't be perfectly synchronized (this is expected and acceptable)
CPU performance: 200-500ms inference time means slight delay (acceptable for demo, could use GPU for production)
Memory usage: Model stays in RAM (~2GB) for lifetime of server process
Browser compatibility: Requires modern browser with WebSocket + MediaDevices support
Network dependency: Requires stable connection to Gladia's API
No persistence: Results are ephemeral (could add recording/export feature later)
📚 Key Resources
Gladia Real-Time API Docs: https://docs.gladia.io/reference/live-audio
FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
Existing model: 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes on HuggingFace
Good luck with the implementation! 🚀