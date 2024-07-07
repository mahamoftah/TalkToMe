import os
from groq import Groq as groq
from streamlit_mic_recorder import speech_to_text

GROQ_API_KEY = "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"


class GroqSTT:
    def __init__(self, model_path="whisper-large-v3",
                 api_key_=GROQ_API_KEY, proxy_url=None):
        self.client = groq(api_key=api_key_)
        self.model = model_path

    def transcribe_audio(self, audio_file, lang):
        with open(audio_file, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model=self.model,
                prompt="Specify context or spelling",
                response_format="json",
                language=lang,
                temperature=0.0
            )
        return transcription.text

    def speech_to_text_streamlit(self, lang):
        text = speech_to_text(
            language=lang,
            start_prompt='🔇',
            stop_prompt="🔈",
            just_once=False,
            use_container_width=False,
            callback=None,
            args=(),
            kwargs={},
            key=None
        )

        return text
