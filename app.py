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

async def text_to_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("question.mp3")

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Question")
    current_q = QUESTIONS[st.session_state.q_index]
    st.write(f"### {current_q}")
    
    # Official Indian & US Voice IDs
    voice_map = {
        "Indian (Male)": "en-IN-PrabhatNeural",
        "Indian (Female)": "en-IN-NeerjaNeural",
        "Andrew (US)": "en-US-AndrewNeural",
        "Emma (US)": "en-US-EmmaNeural"
    }
    voice_choice = st.selectbox("Select TTS Voice", list(voice_map.keys()))
    voice_option = voice_map[voice_choice]
    
    if st.button("🔊 Play Question"):
        asyncio.run(text_to_speech(current_q, voice_option))
        st.audio("question.mp3")

with col2:
    st.subheader("Your Response")
    
    # Setup columns for Radio and the 'i' Info icon
    col2a, col2b = st.columns([10, 1])
    with col2a:
        mode = st.radio("Choose input method:", ["Type Answer", "Record Audio"], horizontal=True)
    with col2b:
        # Native Streamlit hover tooltip
        st.markdown("###", help="""Recording Tips:
1. Click record to start.
2. If unsatisfied, click again to rerecord.
3. Review your text in the read-only box below.
4. Click Save Answer to finalize.""")

    if mode == "Record Audio":
        audio_value = st.audio_input("Record your answer")
        if audio_value:
            with open("answer.wav", "wb") as f:
                f.write(audio_value.read())
            # Transcribe
            result = model.transcribe("answer.wav")
            st.session_state.transcription = result["text"]
        
        st.text_area("Transcription (Review Only)", value=st.session_state.transcription, disabled=True)
    else:
        # Manual typing mode
        st.session_state.transcription = st.text_area("Type your answer", value=st.session_state.transcription)
    
    # Save/Submit Button
    if st.button("💾 Save Answer"):
        if st.session_state.q_index < len(QUESTIONS) - 1:
            st.session_state.q_index += 1
            st.session_state.transcription = ""
            st.rerun()
        else:
            st.success("All questions completed!")