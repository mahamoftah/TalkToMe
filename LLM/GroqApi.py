import os
from groq import Groq as groq
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


class Groq:
    def __init__(self, model_path="llama3-70b-8192", api_key_=GROQ_API_KEY, proxy_url=None):
        self.client = groq(api_key=api_key_)
        self.model = model_path

    def generate(self, question, lang):

        if lang == "ar":
            context = "you are a helpful assistant, Please Answer in Arabic."
        else:
            context = "you are a helpful assistant."

        response = self.client.chat.completions.create(messages=[
            {
                "role": "system",
                "content": context
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": question,
            }
        ],
            model=self.model,
            temperature=0.5,
            max_tokens=1024,
            stream=True,
        )
        for res in response:
            yield res.choices[0].delta.content


def test():
    llm = Groq()

    for answer in llm.generate("who are you"):
        print(answer)

    # sleep(5)
    for answer in llm.generate("who designed you?"):
        print(answer)

    for answer in llm.generate("what are you capable of?"):
        print(answer)


if __name__ == '__main__':
    test()
