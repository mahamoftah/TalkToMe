import streamlit as st
import os
from st_audiorec import st_audiorec
from streamlit_TTS import text_to_speech, text_to_audio
from io import BytesIO
import base64
from gtts import gTTS
from LLM.Embedding import *
from LLM.Gemini import *
from LLM.GroqApi import *
import re
from STT.GroqApiSTT import *

# Define model options
modelOptions = {
    'Gemini': 'gemini-1.5-flash',
    'Gemma2 9b': 'gemma2-9b-it',
    "Gemma 7b": 'gemma-7b-it',
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "LLaMA3 70b": "llama3-70b-8192",
    "LLaMA3 8b": "llama3-8b-8192",
}

# Dropdown menu for model selection
selected_model = st.sidebar.selectbox("Select a model", list(modelOptions.keys()))
selected_model_id = modelOptions[selected_model]

if selected_model_id == 'gemini-1.5-flash':
    chatModel = Gemini()
else:
    chatModel = Groq(model_path=selected_model_id)

STTModel = GroqSTT()
Vectoriser = PDFVectoriser()


languages = ['English', 'Arabic']
language = st.sidebar.selectbox("Select a language", languages)
lang = language[:2].lower()


def play_audio(audio_buffer):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    audio_data = f"data:audio/wav;base64,{audio_base64}"

    audio_html = f"""
    <audio id="audio" autoplay>
        <source src="{audio_data}" type="audio/wav">
    </audio>
    <script>
        var audio = document.getElementById('audio');
        audio.play();
    </script>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# HTML and CSS for styled title
title_html = """
    <style>
    .title {
        font-size: 70px;
        font-weight: 800;
        color: #c13584; /* Gradient color start */
        background: -webkit-linear-gradient(#4c68d7, #ff6464); /* Gradient background */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 50px;
        font-weight: 400;
        color: #333337; /* Subtitle color */
    }
    </style>
    <div class="title">Hello,</div>
    <div class="subtitle">How can I help you today?</div>
    """

st.markdown(title_html, unsafe_allow_html=True)

# Initialize session state variables
if "pdf" not in st.session_state:
    st.session_state.pdf = None
if "v_db" not in st.session_state:
    st.session_state.v_db = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar to switch between text and audio interaction
interaction_mode = st.sidebar.radio("Interaction Mode", ["Text", "Audio"])

# Shared file upload functionality
st.sidebar.title("Chatbot")
pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if pdf and st.sidebar.button("Create Vector Database"):
    with st.spinner("Creating vector database..."):
        texts = Vectoriser.split_text(Vectoriser.extract_from_pdf(pdf))
        if not texts:
            st.error("No texts were extracted from the PDF.")
        else:
            st.session_state.v_db = Vectoriser.create_vector_db(texts)
            st.session_state.pdf = pdf
            st.session_state.texts = texts
            if st.session_state.v_db is not None:
                st.success("Vector database created successfully!")
            else:
                st.error("Failed to create vector database.")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat history cleared!")

if st.sidebar.button("Delete Vector Database"):
    st.session_state.v_db = None
    st.session_state.pdf = None
    st.session_state.texts = None
    st.success("Vector database deleted!")

# Function to process audio input and generate response
# def process_audio(audio_data):
#     with open("recorded_audio.wav", "wb") as f:
#         f.write(audio_data)

#     # Transcribe audio using Whisper
#     transcription = STTModel.transcribe_audio("recorded_audio.wav")
#     st.session_state.messages.append({"role": "user", "content": transcription})

#     # Generate response
#     response = ""
#     for res in STTModel.get_response(transcription):
#         response += res

#     st.session_state.messages.append({"role": "AI", "content": response})
#     return response

if interaction_mode == "Text":
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Enter your message:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        placeholder = st.chat_message("AI").empty()
        similar_text = "You are a Multi Task AI Agent"

        if st.session_state.v_db:
            similar_context = Vectoriser.get_similar_context(user_input, 5, st.session_state.v_db)
            # print("we are here searching")
            # print(similar_context)
            for doc in similar_context:
                similar_text += doc.page_content
                # print(doc)
                # print("simialr text found")

        with st.spinner("Thinking..."):
            stream_res = ""
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            combined_input = f"{conversation_history}\nuser: {user_input}\nAI:"
            combined_input += similar_text
            # print("simialr text added", similar_text)

            for response in chatModel.generate(combined_input, lang):
                if response is None:
                    break
                stream_res += response
                placeholder.markdown(stream_res)
            st.session_state.messages.append({"role": "AI", "content": stream_res})

elif interaction_mode == "Audio":

    st.title("Voice Interaction")

    # Recording audio
    st.subheader("Record Your Message:")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        with open("recorded_audio.wav", "wb") as f:
            f.write(wav_audio_data)

        # # Get the URL for the audio data
        # audio_url = st.audio(wav_audio_data, format="audio/wav")

        # # Use custom HTML and JavaScript to autoplay the audio and make it invisible
        # st.markdown(f"""
        # <audio id="audio" autoplay>
        #     <source src="{audio_url}" type="audio/wav">
        # </audio>
        # <script>
        #     var audio = document.getElementById('audio');
        #     audio.style.display = 'none';
        #     audio.play();
        # </script>
        # """, unsafe_allow_html=True)

        # Transcribe audio using Whisper
        transcription = STTModel.transcribe_audio("recorded_audio.wav", lang)
        # print(transcription)
        user_input = transcription

        st.session_state.messages.append({"role": "user", "content": user_input})
        # st.chat_message("user").write(user_input)
        # placeholder = st.chat_message("AI").empty()
        similar_text = "You are a Multi Task AI Agent"

        if st.session_state.v_db:
            similar_context = Vectoriser.get_similar_context(user_input, 5, st.session_state.v_db)
            for doc in similar_context:
                similar_text += doc.page_content

        with (st.spinner("Thinking...")):
            stream_res = ""
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            combined_input = f"{conversation_history}\nuser: {user_input}\nAI:"
            combined_input += similar_text

            for response in chatModel.generate(combined_input, lang):
                if response is None:
                    break
                stream_res += response
                # placeholder.markdown(stream_res)
            st.session_state.messages.append({"role": "AI", "content": stream_res})

            pattern = re.compile(r'[*#,]')
            text = pattern.sub('', stream_res)

            st.write("Debug")

            if stream_res:
                # Convert text to speech
                sound_file = BytesIO()
                tts = gTTS(text, lang=lang)
                tts.write_to_fp(sound_file)
                play_audio(sound_file)
            else:
                st.warning('No text to convert to speech.')