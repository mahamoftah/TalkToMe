import os
import google.generativeai as genai
import streamlit as st

from time import sleep


def configure_api(api_key, proxy_url=None):
    if proxy_url:
        os.environ['https_proxy'] = proxy_url if proxy_url else None
        os.environ['http_proxy'] = proxy_url if proxy_url else None
    genai.configure(api_key=api_key)




class Gemini:
    def __init__(self, api_key, model_path='gemini-1.5-flash',
                 proxy_url=None):
        configure_api(api_key, proxy_url)
        self.model = genai.GenerativeModel(model_path)

    def generate(self, question, lang):
        response = self.model.generate_content(question, stream=True)
        for res in response:
            if res and hasattr(res, 'text') and res.text:
                yield res.text


def test():
    llm = Gemini()

    for answer in llm.generate("who are you"):
        print(answer)

    # sleep(5)
    for answer in llm.generate("who designed you?"):
        print(answer)

    for answer in llm.generate("what are you capable of?"):
        print(answer)


if __name__ == '__main__':
    test()
