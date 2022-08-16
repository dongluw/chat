import streamlit as st
import cohere
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import random


# set the page title and favicon
st.set_page_config(page_title="Fairy Tale", page_icon="🪄")
# initialize the Cohere SDK
COHERE_API_KEY = "4hV3iRWhB1hUn9pJ0VhMT5VZocEipNlrDD055000"
co = cohere.Client(COHERE_API_KEY)

nltk.download('punkt')
# ask user for name of land they want to generate a fairy tale about
name_of_land = st.text_input("Give me a sentence:", placeholder="Once upon a time in a land called Tieling we came upon a wonderful waterway.")
if name_of_land:
    prompt = word_tokenize(name_of_land)
    random.shuffle(prompt)
    prompt = ["RANDOM:"] + prompt
    prompt = " ".join(prompt)
    # output the generation
    st.header(prompt)

permutated = st.text_input("Give me the permutated sentence:", placeholder="")
if permutated:
    generation = co.generate(
        model='133a5d31-9740-4700-87a2-dbf01a52cfc1-ft',
        prompt=prompt + "\nSORTED:",
        max_tokens=35,
        temperature=0.3,
        stop_sequences=["\n"]).generations[0].text

    st.header("SORTED:" + generation)