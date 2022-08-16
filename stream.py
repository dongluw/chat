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
    st.header("Permutated:")
    random.shuffle(prompt)
    st.header(" ".join(prompt))

temp = st.number_input('Temperature', value=0.3)
permutated = st.text_input("Give me the permutated sentence:", placeholder="")
if permutated:
    generation = co.generate(
        model='0c12477e-d4ef-4eff-bf91-91b491dc6612-ft',
        prompt="RANDOM: " + permutated.strip() + "\nSORT:",
        max_tokens=35,
        temperature=temp,
        stop_sequences=["\n"]).generations[0].text

    st.header("Sorted with temperature " + str(temp) + ":")
    st.header(generation)
