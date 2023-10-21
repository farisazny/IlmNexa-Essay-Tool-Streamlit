import streamlit as st
import PyPDF2
import re
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['OPENAI_API_KEY'] = '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'

def main():
    
    default_text = " "
    st.title(":coffee: IlmNexa")
    st.markdown("<h2 style='text-align: center;'>Your Essay Enhancement Companion 🚀</h2>", unsafe_allow_html=True)

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
 
        
    # Text area for passage input
    user_input = st.text_area("Enter your essay here:  ", value=default_text, height=240)
    
    

    # Checkboxes
    col1, col2, col3 = st.columns(3)
    st.write(" ")
    grammar = col1.checkbox("Check Spelling & Grammar")
    sentiment = col2.checkbox("Check Sentiment")
    paraphrase = col3.checkbox("Paraphrase Essay")

    
    # Button
    if st.button("Submit Essay"):
        st.write("Essay Submitted :smile:")

        if grammar:
            
            correct_grammar(str(user_input))

        if sentiment:
            sentiment_analysis(user_input)

        if paraphrase:
            ptext = paraphrase_text(str(user_input))
            st.write("Paraphrased Text: ", ptext)

        
if __name__ == '__main__':
    main()
