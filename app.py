import streamlit as st
import asyncio
import edge_tts
import whisper
import os

# --- Configuration ---
QUESTIONS = ["What is your name?", "What is your favorite programming language?", "Describe your project."]
model = whisper.load_model("base")

st.set_page_config(layout="wide")

if 'q_index' not in st.session_state: st.session_state.q_index = 0
if 'transcription' not in st.session_state: st.session_state.transcription = ""

# --- Functions ---
async def text_to_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("question.mp3")

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Question")
    current_q = QUESTIONS[st.session_state.q_index]
    st.write(f"### {current_q}")
    
    # Voice Selection
    voice_option = st.selectbox("Select Voice", 
                                ["en-US-ChristopherNeural", "en-US-BrianNeural", "en-US-AndrewNeural"])
    
    if st.button("🔊 Play Question"):
        asyncio.run(text_to_speech(current_q, voice_option))
        st.audio("question.mp3")

with col2:
    st.subheader("Your Response")
    # Toggle between Typing and Recording
    mode = st.radio("Choose input method:", ["Type Answer", "Record Audio"])
    
    if mode == "Record Audio":
        audio_value = st.audio_input("Record your answer")
        if audio_value:
            with open("answer.wav", "wb") as f:
                f.write(audio_value.read())
            result = model.transcribe("answer.wav")
            st.session_state.transcription = result["text"]
        
        st.text_area("Transcription (Review Only)", value=st.session_state.transcription, disabled=True)
    
    else:
        # Manual typing mode
        st.session_state.transcription = st.text_area("Type your answer", value=st.session_state.transcription)
    
    if st.button("Submit Answer"):
        if st.session_state.q_index < len(QUESTIONS) - 1:
            st.session_state.q_index += 1
            st.session_state.transcription = ""
            st.rerun()
        else:
            st.success("All questions completed!")
