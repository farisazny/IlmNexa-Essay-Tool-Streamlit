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

def main():
    
    os.environ['OPENAI_API_KEY'] = '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'


    st.title("🤖 IlmNexa Bot ☕️")
    st.write("Welcome to IlmNexa Bot! Your friendly essay enhancement companion.")

    default_text = " "
    with st.expander("Click to enter your essay"):
        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        # Display the uploaded file
        if uploaded_file is not None:
            default_text = extract_text_from_pdf(uploaded_file)

            for text in default_text:
                split_message = re.split(r'\s+|[ ]\s*', text)
                new_text = ' '.join(split_message)
            default_text = new_text
            user_input = new_text
            st.write("Uploaded filename:", uploaded_file.name)
            
        
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = np.array(cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1))
            
            
            # Save the image
            cv2.imwrite("uploaded_image.jpg", image)
            
            # Extract text from the image
            default_text = extract_text_from_image(image, uploaded_image.name)
            user_input = default_text
        prompt = st.text_area("Enter your essay here:", value=default_text, height=240)
    
    # Selectbox
    options = ["Argumentative","Persuasive", "Descriptive", "News Report", "Tutorial"]
    selected_option = st.selectbox("Select an essay type", options)
        
    

     # Slider
    formal = st.slider("Rate on how FORMAL this document is supposed to be (0-5)", 0, 5)
    fun = st.slider("Rate on how CREATIVE this document is supposed to be (0-5)", 0, 5)
        
    # To feed how creative the GPT should be
    temp = calc_llm_temp(formal, fun)

    # Checkbox options for user selection
    col1, col2, col3, col4 = st.columns(4)
    st.write(" ")
    suggestion = col1.checkbox("Title Suggestions")
    brainstorm = col2.checkbox("Brainstorm ideas")
    grading = col3.checkbox("Essay Grading")
    feedback = col4.checkbox("Essay Feedback")

    essay_type = selected_option

    if st.button("Submit"):
        if suggestion:
            gpt_turbo(
                prompt,
                "Title Suggestion",
                "I can help you coming up with a title! Please input your essay below",
                """I'm brainstorming essay ideas for my school assignment. Write me 5 
                    title ideas about this subject: {topic}.
                    Keep in mind that the essay type is: {essay_type}.
                    Write one point in one line.
                    Make sure the text you write is black in color.""",
                temp
            )

        if brainstorm:
            gpt_turbo(
                prompt,
                "Brainstorm Idea",
                "Give us a topic to brainstorm on",
                """Brainstorm some ideas and turn it into 5 number points for this topic: {topic}.
                Keep in mind that the essay type is: {essay_type}.
                Write one point in one line.
                Make sure the text you write is black in color.""",
                temp
            )

        if grading:
            gpt_turbo(
                prompt,
                "AI Essay Grading",
                "Give us an essay and we will help you grade it!",
                """
                I'm going to give you an essay and please grade it from 0-100: {topic}.
                Keep in mind that the essay type is: {essay_type}.
                Give some evaluations on the writing and what could've been better.
                You should answer with 1. Final Grade 2. Evaluations.
                Grade and Evaluations should not be on the same line.
                Make sure the text you write is black in color.
                """,
                temp
            )

        if feedback:
            gpt_turbo(
                prompt,
                "AI Essay Feedback",
                "Give us an essay and we will help you grade it!",
                """I'm going to give you a {essay_type} essay: {topic}.
                Give me 5 number points on how to make this essay better.
                Each number point should be on different lines.
                Make sure the text you write is black in color.
                """,
                temp
            )

if __name__ == '__main__':
    main()