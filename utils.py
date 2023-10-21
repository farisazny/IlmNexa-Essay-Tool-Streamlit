import streamlit as st
import PyPDF2
import re
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from gingerit.gingerit import GingerIt
import pandas as pd
import matplotlib.pyplot as plt
import os
#import easyocr
import pytesseract
import cv2
import numpy as np
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
nltk.download('wordnet')
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from transformers import pipeline


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def paraphrase_text(text):
    st.markdown("<h2 style='text-align: center;'>Paraphrasing</h2>", unsafe_allow_html=True)
    tokens = word_tokenize(text)
    paraphrased_text = []

    count = 0
    for token in tokens:
        if count % 5 == 0:
            synonyms = get_synonyms(token)
            if synonyms:
                paraphrased_text.append(random.choice(synonyms))
            else:
                paraphrased_text.append(token)
        else:
            paraphrased_text.append(token)

        count += 1

    return ' '.join(paraphrased_text)


def extract_text_from_image(image, image_name):
    
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text
    

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        raise ValueError("PDF file is not provided.")
    
    # Open the PDF file of your choice
    
    reader = PyPDF2.PdfReader(pdf_file, strict=False)
    pdf_text = []

    for page in reader.pages:
        content = page.extract_text()
        pdf_text.append(content)

    return pdf_text
    
def calc_llm_temp(formal, fun):
    temp = 5
    temp = temp - formal + fun
    temp = temp/10
    return temp

def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    st.markdown("<h2 style='text-align: center;'>Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.write("Negative: ", sentiment_scores['neg'])
    st.write("Positive: ", sentiment_scores['pos'])
    st.write("Neutral: ", sentiment_scores['neu'])
    st.write("Compound: ", sentiment_scores['compound'])

    # Create a DataFrame with sentiment scores
    data = pd.DataFrame({'Sentiment': ['Negative', 'Positive', 'Neutral', 'Compound'],
                         'Score': [sentiment_scores['neg'], sentiment_scores['pos'],
                                   sentiment_scores['neu'], sentiment_scores['compound']]})

    
    # Define colors for each sentiment category
    colors = ['#FF6961', '#7FFF7F', '#808080']

    # Determine color for compound score
    compound_color = colors[1] if sentiment_scores['compound'] >= 0 else colors[0]

    # Append compound color to the colors list
    colors.append(compound_color)

    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(data['Sentiment'], data['Score'], color=colors)
    ax.set_xlabel('Score')
    ax.set_ylabel('Sentiment')
    ax.set_title('Sentiment Analysis')
    ax.set_xlim(-1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(False)

    # Display the plot using Streamlit
    st.pyplot(fig)



def correct_spell(text):
    spell_check = TextBlob(" ")
    words = text.split()
    corrected_words = []
    for word in words:
        corrected_word = str(TextBlob(word).correct())
        corrected_words.append(corrected_word)
    st.write("Corrected Words: ", " ".join(corrected_words))
    

def correct_grammar(text):

    st.markdown("<h2 style='text-align: center;'>Grammar Mistakes</h2>", unsafe_allow_html=True)
    if len(text) > 299:
        
        st.warning("GingerIT API does not support more than 300 characters")
    else:
        grammar_check = GingerIt()
        matches = grammar_check.parse(text)
        
        foundmistakes = []
        for error in matches['corrections']:
            foundmistakes.append(error['text'])
        foundmistakes_count = len(foundmistakes)

        
        st.write("Mistakes: ", ",  ".join(foundmistakes))
        st.write("Mistakes Count: ", foundmistakes_count)

def gpt_turbo(prompt, gpt_title, title_text, prompt_template, temp):
    os.environ['OPENAI_API_KEY'] = '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'

    st.header(gpt_title)
    #Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic', 'essay_type'],
        template = prompt_template
    )

    prompt = prompt
    llm = OpenAI(temperature=temp)
    title_chain = LLMChain(llm = llm, prompt = title_template)

    if prompt: 
        response = title_chain.run(topic= prompt, essay_type =prompt)
        st.write(response)
        
def img_to_text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text
