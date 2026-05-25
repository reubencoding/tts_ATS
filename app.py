import streamlit as st
import asyncio
import edge_tts
import whisper

# --- Configuration ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()
QUESTIONS = ["What is your name?", "What is your favorite programming language?", "Describe your project."]

st.set_page_config(layout="wide")

# State Management
if 'q_index' not in st.session_state: st.session_state.q_index = 0
if 'transcription' not in st.session_state: st.session_state.transcription = ""
if 'text_input' not in st.session_state: st.session_state.text_input = ""
if 'audio_file' not in st.session_state: st.session_state.audio_file = None

async def text_to_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("question.mp3")

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Question")
    current_q = QUESTIONS[st.session_state.q_index]
    st.write(f"### {current_q}")
    
    voice_map = {
        "Indian (Male)": "en-IN-PrabhatNeural",
        "Indian (Female)": "en-IN-NeerjaNeural",
        "Andrew (US)": "en-US-AndrewNeural",
        "Emma (US)": "en-US-EmmaNeural"
    }
    voice_choice = st.selectbox("Select TTS Voice", list(voice_map.keys()))
    
    if st.button("🔊 Play Question"):
        asyncio.run(text_to_speech(current_q, voice_map[voice_choice]))
        st.audio("question.mp3")

with col2:
    st.subheader("Your Response")
    
    col2a, col2b = st.columns([10, 1])
    with col2a:
        mode = st.radio("Choose input method:", ["Type Answer", "Record Audio"], horizontal=True)
    with col2b:
        st.markdown("### ", help="""Recording Tips:
1. Click record to start.
2. If unsatisfied, click again to rerecord.
3. Your transcript will appear in the read-only box below.
4. Click Save Answer to finalize.""")

    if mode == "Record Audio":
        # Capture audio and persist in session
        audio_value = st.audio_input("Record your answer")
        if audio_value:
            st.session_state.audio_file = audio_value.read()
            with open("answer.wav", "wb") as f:
                f.write(st.session_state.audio_file)
            result = model.transcribe("answer.wav")
            st.session_state.transcription = result["text"]
        
        # Display playback if audio exists
        if st.session_state.audio_file:
            st.audio(st.session_state.audio_file)
            st.text_area("Transcription (Review Only)", value=st.session_state.transcription, disabled=True)
    
    else:
        # Manual typing mode uses a separate session state
        st.session_state.text_input = st.text_area("Type your answer", value=st.session_state.text_input)
    
    if st.button("💾 Save Answer"):
        if st.session_state.q_index < len(QUESTIONS) - 1:
            st.session_state.q_index += 1
            # Reset state for next question
            st.session_state.transcription = ""
            st.session_state.text_input = ""
            st.session_state.audio_file = None
            st.rerun()
        else:
            st.success("All questions completed!")