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
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

HUGGINGFACEHUB_API_TOKEN = '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'

def main():
    
    os.environ['OPENAI_API_KEY'] = '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'


    st.title("🖼️ IlmNexa Tagger ☕️")
    st.write("Upload an image and we'll give you ideas on what to write on the subject!")

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
    
    st.write("""Unfortunately, this device does not have enough GPU memory to load the model.
             You can find the PEFT-LORA Fine-Tuning demo here:
             https://colab.research.google.com/drive/1Q4iY4FoY2Mk4x_smuQOiIldpdB4Una_Z?usp=sharing """)

    # peft_model_id = "FarisAzny/bloom-1b1-lora-tagger"
    # config = PeftConfig.from_pretrained(peft_model_id)
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
    #                                             return_dict=True, load_in_8bit=True, 
                                                
    #                                             device_map = 'cuda'
    #                                             ) # Use CPU for the model and GPU for the 'module' module 
    # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # # Load the Lora model
    # model = PeftModel.from_pretrained(model, peft_model_id)


if __name__ == '__main__':
    main()