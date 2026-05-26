import streamlit as st
import numpy as np
import wave
import json
import asyncio
import edge_tts
import os
import vosk
import time
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pydub import AudioSegment

# Suppress verbose Vosk/Kaldi logging in terminal
vosk.SetLogLevel(-1)

# --- Configuration & State ---
QUESTIONS = [
    "What is your name?",
    "What is your favorite programming language?",
    "Describe your project."
]

VERIFICATION_PHRASES = [
    "Vocal fingerprint security code is alpha four nine seven",
    "Audio authorization sequence completed for candidate identity check",
    "Confirm speaking pattern matching baseline biometric verification test",
    "Random candidate authorization code seven five three zero"
]

st.set_page_config(
    page_title="Secure Mock Interview Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Initialize Session States
if 'q_index' not in st.session_state:
    st.session_state.q_index = 0
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'speaker_status' not in st.session_state:
    st.session_state.speaker_status = None
if 'registered_speaker' not in st.session_state:
    st.session_state.registered_speaker = None

# Advanced Analytics Session States
if 'question_load_time' not in st.session_state:
    st.session_state.question_load_time = time.time()
if 'question_stalling_latency' not in st.session_state:
    st.session_state.question_stalling_latency = 0.0
if 'current_wpm' not in st.session_state:
    st.session_state.current_wpm = 0
if 'current_gaps' not in st.session_state:
    st.session_state.current_gaps = []
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
if 'verification_status' not in st.session_state:
    st.session_state.verification_status = "not_started"  # not_started, passed, failed
if 'verification_phrase' not in st.session_state:
    st.session_state.verification_phrase = random.choice(VERIFICATION_PHRASES)

# --- Resource Loading ---
@st.cache_resource
def load_vosk_models():
    """
    Downloads and caches Vosk translation and speaker models.
    Uses cached directories automatically if already downloaded.
    """
    with st.spinner("Initializing AI Models (Offline Vosk & Speaker Verification)..."):
        # Auto-downloads small English transcription model (approx. 40MB)
        model = vosk.Model(lang="en-us")
        # Auto-downloads speaker identification model (approx. 13MB)
        spk_model_path = model.get_model_path("vosk-model-spk-0.4", None)
        spk_model = vosk.SpkModel(spk_model_path)
    return model, spk_model

vosk_model, spk_model = load_vosk_models()

# --- Audio Processing Helpers ---
async def text_to_speech(text, voice):
    """Generates audio for the interview question using edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("question.mp3")

def process_audio(audio_bytes):
    """
    Saves raw audio bytes, converts to 16kHz mono WAV,
    and performs transcription, word timestamp analysis, and speaker embedding in a single pass.
    """
    raw_path = "temp_raw.wav"
    with open(raw_path, "wb") as f:
        f.write(audio_bytes)
        
    try:
        audio = AudioSegment.from_file(raw_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        wav_path = "temp_processed.wav"
        audio.export(wav_path, format="wav")
    except Exception as e:
        st.error(f"Error decoding audio format: {e}")
        return "", np.array([]), 0, []
        
    try:
        wf = wave.open(wav_path, "rb")
        rec = vosk.KaldiRecognizer(vosk_model, 16000)
        rec.SetSpkModel(spk_model)
        # Enable word-level timestamps output
        rec.SetWords(True)
        
        vectors = []
        transcript_parts = []
        words_data = []
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if 'text' in res and res['text'].strip():
                    transcript_parts.append(res['text'])
                if 'result' in res:
                    words_data.extend(res['result'])
                if 'spk' in res:
                    vectors.append(res['spk'])
                    
        # Grab final results
        res = json.loads(rec.FinalResult())
        if 'text' in res and res['text'].strip():
            transcript_parts.append(res['text'])
        if 'result' in res:
            words_data.extend(res['result'])
        if 'spk' in res:
            vectors.append(res['spk'])
            
        # Clean up files
        wf.close()
        for p in [raw_path, wav_path]:
            if os.path.exists(p):
                os.remove(p)
                
        full_transcript = " ".join(transcript_parts).strip()
        
        # --- Chunked Speaker Vector Extraction ---
        chunk_vectors = []
        chunk_length_ms = 3000  # 3 seconds
        audio_len_ms = len(audio)
        
        for i in range(0, audio_len_ms, chunk_length_ms):
            chunk = audio[i:i+chunk_length_ms]
            if len(chunk) < 1500:
                continue
            
            # Extract raw 16kHz mono bytes from the chunk
            chunk_bytes = chunk.raw_data
            
            chunk_rec = vosk.KaldiRecognizer(vosk_model, 16000)
            chunk_rec.SetSpkModel(spk_model)
            chunk_rec.AcceptWaveform(chunk_bytes)
            
            chunk_res = json.loads(chunk_rec.FinalResult())
            if 'spk' in chunk_res:
                # Ensure the chunk contains active speech
                if 'text' in chunk_res and chunk_res['text'].strip():
                    chunk_vectors.append(chunk_res['spk'])
                    
        # Failsafe: Fall back to continuous loop vectors if no chunk-level vectors were detected
        final_vectors = chunk_vectors if len(chunk_vectors) > 0 else vectors
        
        # --- Advanced Analytics Extraction ---
        wpm = 0
        thought_gaps = []
        
        if len(words_data) >= 1:
            # Speaking Rate WPM
            total_duration = words_data[-1]['end'] - words_data[0]['start']
            wpm = int(len(words_data) / (total_duration / 60)) if total_duration > 0 else 0
            
            # Thought Gaps (pauses > 1.5 seconds)
            for i in range(1, len(words_data)):
                gap = words_data[i]['start'] - words_data[i-1]['end']
                if gap > 1.5:
                    thought_gaps.append({
                        "word_before": words_data[i-1]['word'],
                        "word_after": words_data[i]['word'],
                        "start_time": round(words_data[i-1]['end'], 2),
                        "duration": round(gap, 2)
                    })
                    
        return full_transcript, np.array(final_vectors), wpm, thought_gaps
    except Exception as e:
        st.error(f"Error during Speech-to-Text: {e}")
        return "", np.array([]), 0, []

def save_interview_logs_to_database():
    """Saves the candidate metrics history and verification results locally as our database."""
    log_data = {
        "interview_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verification_status": st.session_state.verification_status,
        "total_questions": len(QUESTIONS),
        "metrics_history": st.session_state.metrics_history
    }
    try:
        with open("interview_logs.json", "w") as f:
            json.dump(log_data, f, indent=4)
    except Exception as e:
        st.error(f"Database Log Error: Could not save logs: {e}")

# --- UI Header ---
st.title("🎙️ Secure Mock Interview & Speaker Verifier")
st.markdown(
    "Practice your interview offline while ensuring identity security via speech biometrics."
)
st.write("---")

# --- UI Flow Control ---
if st.session_state.q_index >= len(QUESTIONS):
    # --- PHASE 2: Mandatory Voice Verification Challenge & Dashboard ---
    if st.session_state.verification_status == "not_started":
        st.subheader("👤 Mandatory Voice Verification Challenge")
        st.markdown(
            """Before your interview is formally completed, you must complete a secure identity verification challenge. 
This ensures your vocal acoustics match the registered candidate voice baseline profile."""
        )
        
        # Display instruction card
        st.warning(
            f"**Please record yourself reading the following phrase exactly:**\n\n### \"{st.session_state.verification_phrase}\""
        )
        
        audio_challenge = st.audio_input("Record verification code")
        
        if audio_challenge:
            if st.button("Submit Verification Check", use_container_width=True):
                with st.spinner("Verifying identity acoustics and phrase consistency..."):
                    transcript, vectors, wpm, thought_gaps = process_audio(audio_challenge.read())
                    
                    # 1. Text Phrase Verification Check
                    phrase_words = [w.lower().strip(",.!?\"'") for w in st.session_state.verification_phrase.split() if len(w) > 3]
                    matching_words = [w for w in phrase_words if w in transcript.lower()]
                    phrase_match_ratio = len(matching_words) / len(phrase_words) if phrase_words else 1.0
                    phrase_passed = phrase_match_ratio >= 0.70
                    
                    # 2. Voice Print Matching Check
                    voice_passed = False
                    voice_similarity = 0.0
                    
                    if len(vectors) >= 2 and st.session_state.registered_speaker is not None:
                        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1e-6, norms)
                        vectors_norm = vectors / norms
                        
                        current_profile = np.mean(vectors_norm, axis=0)
                        current_profile = current_profile / np.linalg.norm(current_profile)
                        
                        ref_profile = st.session_state.registered_speaker
                        voice_similarity = np.dot(ref_profile, current_profile)
                        voice_passed = voice_similarity >= 0.65
                    
                    # Final Authentication Verdict
                    if phrase_passed and voice_passed:
                        st.session_state.verification_status = "passed"
                        save_interview_logs_to_database()
                        st.balloons()
                        st.rerun()
                    else:
                        st.session_state.verification_status = "failed"
                        st.error(
                            f"""❌ Identity Verification Failed! 
- Phrase Spoken Match: {"PASSED" if phrase_passed else "FAILED"} (Spoken: '{transcript}')
- Acoustic Match Score: {voice_similarity:.2f} ({"PASSED" if voice_passed else "FAILED"} - Must be >= 0.65)
"""
                        )
                        st.write("Please try again ensuring a clear voice match and reading the correct code.")
                        if st.button("Reset Verification Challenge"):
                            st.session_state.verification_status = "not_started"
                            st.rerun()

    elif st.session_state.verification_status == "failed":
        st.subheader("👤 Identity Verification Status: Unverified")
        st.error(
            "🔴 Access Blocked: Security biometrics or code verification did not match the registered candidate."
        )
        if st.button("Try Verification Again"):
            st.session_state.verification_status = "not_started"
            st.rerun()
            
    elif st.session_state.verification_status == "passed":
        st.subheader("👤 Identity Verification Status: Verified Certificate")
        st.success("🎉 Candidate acoustics authenticated! Mock interview successfully verified.")
        
        # --- Premium Analytics Dashboard ---
        st.write("---")
        st.markdown("### 📊 Premium Candidate Performance & Timing Dashboard")
        
        # Render Metrics Grid
        col_m1, col_m2, col_m3 = st.columns(3)
        
        all_latency = [m["stalling_latency_seconds"] for m in st.session_state.metrics_history]
        all_wpm = [m["wpm"] for m in st.session_state.metrics_history if m["wpm"] > 0]
        all_gaps = sum([m["thought_gaps_count"] for m in st.session_state.metrics_history])
        
        avg_latency = np.mean(all_latency) if all_latency else 0
        avg_wpm = np.mean(all_wpm) if all_wpm else 0
        
        col_m1.metric("Average Stalling Delay", f"{avg_latency:.1f} sec", help="Latency between loading question and commencing response.")
        col_m2.metric("Average Speech WPM", f"{int(avg_wpm)} WPM", help="Speech articulation rate (Words Per Minute).")
        col_m3.metric("Total Hesitation Pauses", f"{all_gaps} gaps", help="Significant thought gaps (>1.5s) registered.")
        
        # Render Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("#### ⏳ Stalling Delay (Latency to Start Response)")
            st.bar_chart(
                data={
                    "Question Index": [f"Q{m['question_index'] + 1}" for m in st.session_state.metrics_history],
                    "Stalling Latency (s)": all_latency
                },
                x="Question Index",
                y="Stalling Latency (s)"
            )
            
        with col_c2:
            st.write("#### 🎙️ Speaking Rate Trend (WPM)")
            st.line_chart(
                data={
                    "Question Index": [f"Q{m['question_index'] + 1}" for m in st.session_state.metrics_history],
                    "Speech WPM": [m["wpm"] for m in st.session_state.metrics_history]
                },
                x="Question Index",
                y="Speech WPM"
            )
            
        # Thought Gaps Details Table
        st.write("---")
        st.write("#### 🔍 Hesitations and Thought Gaps Details")
        gap_records = []
        for m in st.session_state.metrics_history:
            for gap in m["thought_gaps"]:
                gap_records.append({
                    "Question": f"Q{m['question_index'] + 1}",
                    "Pause Location": f"Between '{gap['word_before']}' and '{gap['word_after']}'",
                    "Acoustic Time (s)": gap["start_time"],
                    "Duration (s)": gap["duration"]
                })
        
        if gap_records:
            st.table(gap_records)
        else:
            st.info("Excellent! No significant thought gaps or pauses (>1.5s) were detected during your responses.")
            
        # Restart Interview
        st.write("---")
        if st.button("🔄 Restart New Interview Session"):
            st.session_state.q_index = 0
            st.session_state.transcription = ""
            st.session_state.text_input = ""
            st.session_state.audio_file = None
            st.session_state.speaker_status = None
            st.session_state.registered_speaker = None
            st.session_state.verification_status = "not_started"
            st.session_state.metrics_history = []
            st.rerun()

else:
    # --- PHASE 1: Standard Mock Interview Flow ---
    col1, col2 = st.columns([1, 1], gap="large")

    # --- Left Column: Question & Playback ---
    with col1:
        st.subheader("Question")
        current_q = QUESTIONS[st.session_state.q_index]
        
        # Display question card
        st.info(f"**Question {st.session_state.q_index + 1} of {len(QUESTIONS)}**:\n\n### {current_q}")
        
        # Voice configurations for TTS
        voice_map = {
            "Indian (Male)": "en-IN-PrabhatNeural",
            "Indian (Female)": "en-IN-NeerjaNeural",
            "Andrew (US - Male)": "en-US-AndrewNeural",
            "Emma (US - Female)": "en-US-EmmaNeural"
        }
        voice_choice = st.selectbox("Select TTS Voice Option", list(voice_map.keys()))
        
        if st.button("🔊 Play Question", use_container_width=True):
            with st.spinner("Synthesizing voice..."):
                asyncio.run(text_to_speech(current_q, voice_map[voice_choice]))
            st.audio("question.mp3", autoplay=True)

        # Show voice enrollment badge if available
        if st.session_state.registered_speaker is not None:
            st.success("👤 Voice Enrollment Active: Candidate voice profile successfully registered.")

    # --- Right Column: User Response & Analysis ---
    with col2:
        st.subheader("Your Response")
        
        # Select input method
        mode = st.radio("Choose input method:", ["Type Answer", "Record Audio"], horizontal=True)
        
        if mode == "Record Audio":
            audio_value = st.audio_input("Record your response")
            
            if audio_value:
                # Read audio bytes
                audio_bytes = audio_value.read()
                
                # Check if this is a new recording
                if st.session_state.audio_file != audio_bytes:
                    st.session_state.audio_file = audio_bytes
                    
                    # Capture stalling latency (first action)
                    st.session_state.question_stalling_latency = round(time.time() - st.session_state.question_load_time, 2)
                    
                    with st.spinner("Analyzing audio response..."):
                        # Extract transcription, speaker vectors, WPM and thought gaps
                        transcript, vectors, wpm, thought_gaps = process_audio(audio_bytes)
                        st.session_state.transcription = transcript
                        st.session_state.current_wpm = wpm
                        st.session_state.current_gaps = thought_gaps
                        
                        # Perform speaker verification using robust centroid-based Cosine Similarity
                        if len(vectors) < 2:
                            st.session_state.speaker_status = {
                                "status": "info",
                                "message": f"Audio response too short to perform reliable speaker verification. WPM: {wpm}."
                            }
                        else:
                            try:
                                # Normalize vectors for cosine-distance calculation
                                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                                norms = np.where(norms == 0, 1e-6, norms)
                                vectors_norm = vectors / norms
                                
                                # Fit KMeans to split into 2 clusters
                                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(vectors_norm)
                                labels = kmeans.labels_
                                
                                if np.sum(labels == 0) > 0 and np.sum(labels == 1) > 0:
                                    # Extract cluster centroids
                                    c1 = np.mean(vectors_norm[labels == 0], axis=0)
                                    c2 = np.mean(vectors_norm[labels == 1], axis=0)
                                    
                                    # Normalize centroids
                                    c1 = c1 / np.linalg.norm(c1) if np.linalg.norm(c1) > 0 else c1
                                    c2 = c2 / np.linalg.norm(c2) if np.linalg.norm(c2) > 0 else c2
                                    
                                    # Calculate cosine similarity between cluster centroids
                                    centroid_sim = np.dot(c1, c2)
                                else:
                                    centroid_sim = 1.0
                                
                                # 1. Check for Intra-Question proxy (multiple speakers in current answer)
                                # A voice similarity threshold of 0.65 separates same vs different speaker
                                if centroid_sim < 0.65:
                                    st.session_state.speaker_status = {
                                        "status": "warning",
                                        "message": f"⚠️ Alert: Multiple speakers detected within this response (Voice similarity: {centroid_sim:.2f}). Potential proxy interview."
                                    }
                                else:
                                    # 2. Check for Inter-Question proxy (speaker swapping between questions)
                                    current_profile = np.mean(vectors_norm, axis=0)
                                    current_profile = current_profile / np.linalg.norm(current_profile)
                                    
                                    if st.session_state.registered_speaker is None:
                                        # First question baseline enrollment
                                        st.session_state.registered_speaker = current_profile
                                        st.session_state.speaker_status = {
                                            "status": "success",
                                            "message": f"✅ Single speaker confirmed (Intra-voice similarity: {centroid_sim:.2f}). Vocal profile registered. Speech speed: {wpm} WPM."
                                        }
                                    else:
                                        # Subsequent questions similarity validation
                                        ref_profile = st.session_state.registered_speaker
                                        voice_similarity = np.dot(ref_profile, current_profile)
                                        
                                        # Threshold 0.65 separates same vs different speaker in Vosk embedding space
                                        if voice_similarity < 0.65:
                                            st.session_state.speaker_status = {
                                                "status": "warning",
                                                "message": f"⚠️ Alert: Voice does not match the enrolled candidate (Voice match score: {voice_similarity:.2f}). Potential proxy speaker swap!"
                                            }
                                        else:
                                            st.session_state.speaker_status = {
                                                "status": "success",
                                                "message": f"✅ Single speaker confirmed. Voice matches candidate (Voice match score: {voice_similarity:.2f}). Speech speed: {wpm} WPM."
                                            }
                            except Exception as e:
                                st.session_state.speaker_status = {
                                    "status": "info",
                                    "message": f"Speaker verification processed successfully. {e}"
                                }
            
            # Display results if present
            if st.session_state.audio_file:
                st.audio(st.session_state.audio_file)
                st.text_area("Live Transcript (Vosk speech-to-text)", value=st.session_state.transcription, height=120, disabled=True)
                
                # Show speaker verification alerts
                if st.session_state.speaker_status:
                    stat = st.session_state.speaker_status
                    if stat["status"] == "warning":
                        st.warning(stat["message"])
                    elif stat["status"] == "success":
                        st.success(stat["message"])
                    else:
                        st.info(stat["message"])
                        
                # Present WPM & Pauses metrics
                if st.session_state.current_wpm > 0:
                    st.write(f"📊 **Speaking Rate**: {st.session_state.current_wpm} WPM")
                    if st.session_state.current_gaps:
                        st.warning(f"⚠️ **Pauses Detected**: {len(st.session_state.current_gaps)} long thought gaps identified (>1.5s).")
                        
        else:
            # Typing mode
            st.session_state.text_input = st.text_area("Type your response here:", value=st.session_state.text_input, height=180)
            
        # Navigation
        st.write("")
        if st.button("💾 Save Answer & Next Question", use_container_width=True):
            # Check that user entered an answer
            has_answered = False
            if mode == "Record Audio" and st.session_state.transcription:
                has_answered = True
            elif mode == "Type Answer" and st.session_state.text_input.strip():
                has_answered = True
                # Record typing mode latency
                st.session_state.question_stalling_latency = round(time.time() - st.session_state.question_load_time, 2)
                st.session_state.current_wpm = 0
                st.session_state.current_gaps = []
                
            if not has_answered:
                st.error("Please enter a response before saving.")
            else:
                # 1. Log metrics to session history for the database
                st.session_state.metrics_history.append({
                    "question_index": st.session_state.q_index,
                    "question": current_q,
                    "input_method": mode,
                    "stalling_latency_seconds": st.session_state.question_stalling_latency,
                    "wpm": st.session_state.current_wpm,
                    "thought_gaps_count": len(st.session_state.current_gaps),
                    "thought_gaps": st.session_state.current_gaps
                })
                
                # 2. Advance to next question
                st.session_state.q_index += 1
                
                # 3. Reset states for next question
                st.session_state.transcription = ""
                st.session_state.text_input = ""
                st.session_state.audio_file = None
                st.session_state.speaker_status = None
                st.session_state.question_load_time = time.time()  # Reset question load timer
                st.rerun()