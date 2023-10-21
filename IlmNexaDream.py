import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
import sys
sys.path.append("..")  # Add the parent folder to the Python import path
from utils import *  # Import the function from main.py
from transformers import pipeline

HUGGINGFACEHUB_API_TOKEN = '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'

def main():
    
    os.environ['OPENAI_API_KEY'] = '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'


    st.title("🖼️ IlmNexa Dream ☕️")
    st.write("Upload an image and we'll give you ideas on what to write on the subject!")

    default_text = " "

    # Selectbox
    options = ["Argumentative","Persuasive", "Descriptive", "News Report", "Tutorial"]
    essay_type = st.selectbox("Select an essay type", options)

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image)
        image = np.array(cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1))
            
            
        # Save the image
        cv2.imwrite("uploaded_image.jpg", image)
        

        caption = img_to_text("uploaded_image.jpg")

        st.write("Caption: ")
        st.header(caption)

        

        topic = caption
        prompt = caption
        temp = 0.9
        # gpt_turbo(
        #             prompt,
        #             "Brainstorm Idea",
        #             "Give us a topic to brainstorm on",
        #             """Brainstorm some ideas and turn it into a mini-essay regarding this: {topic}.
        #             Keep in mind that the essay type is: {essay_type}.
        #             """,
        #             temp
        #         )

        if st.button("Submit"):
        
            gpt_turbo(
                prompt,
                "Brainstorm Ideas",
                "I will help you to think of ideas regarding this image!",
                """I'm brainstorming essay ideas for my school assignment. Write me 5 
                    title ideas about this subject: {topic}.
                    Keep in mind that the essay type is: {essay_type}.
                    Write one point in one line.
                    Make sure the text you write is black in color.""",
                temp
            )

            gpt_turbo(
                prompt,
                "Example Essay",
                "I will help you to think of ideas regarding this image!",
                """Write me a paragraph about:  {topic}.
                    Keep in mind that the essay type is: {essay_type}.
                    Write one point in one line.
                    Make sure the text you write is black in color.""",
                temp
            )

if __name__ == '__main__':
    main()