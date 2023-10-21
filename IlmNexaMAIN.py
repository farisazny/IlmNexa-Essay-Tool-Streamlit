import streamlit as st
from streamlit_option_menu import option_menu



# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 3

def streamlit_menu(example=1):

    if example == 3:
        # 2. horizontal menu with custom style
        st.markdown(
            """
            <style>
            .st-cc {
                max-width: 100%;
                margin: 0 auto;
                padding: 0px;
            }
            .st-ai {
                width: 100%;
                max-width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Pic-To-Words", "Essay Tool", "Study Buddy", "Note Taking"],  # required
            icons=["house", "book", "envelope", "book", "pencil"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "12px"},
                "nav-link": {
                    "font-size": "18px",  # Adjust font size as needed
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)




if selected == "Home":
    import streamlit as st

    def main():
        
        st.image("Main Page.jpg")
        st.image("Main Page 2.jpg")
        st.title("☕️ About IlmNexa")

        st.markdown(
        """
        EspressoWrite is a web application that helps enhance your essays and provide valuable feedback. 
        It offers the following features:

        | Library/API          | Functionality                          | Use                                                |
        | -------------------- | -------------------------------------- | -------------------------------------------------- |
        | Streamlit            | Building the app                        | Create the user interface for EspressoWrite        |
        | Pytesseract          | Image to Text                           | Extract text from images                           |
        | PyPDF2               | PDF to Text                             | Extract text from PDF files                        |
        | NLTK Sentiment       | Sentiment Analysis                      | Analyze the sentiment of essays                    |
        | TextBlob             | Check Spellings                         | Correct spelling mistakes                          |
        | GingerIt             | Check Grammar                           | Identify and correct grammar mistakes              |
        | LangChain            | LLM Workflow                            | Calculate a temperature for the AI text generation |
        | OpenAI GPT           | Essay Feedback + Suggestions            | Generate AI-based feedback and title suggestions   |
        | Bloom 1b Fine-tuning | Custom Language Model Training          | Fine-tuned Bloom 1b model                          |
        | BLIP                 | Image Captioning                        | Generate captions for images                       |
        | PEFT                 | Parameter Efficient Fine Tuning         | Efficient fine-tuning of language models           |
        | LORA                 | Low Rank Adaptation for LLMs            | Adapt language models with low-rank modifications  |
        """
        )


        st.write(" ")
        st.write(" ")
        st.markdown(""" To get started, upload a PDF file or an image containing the text of your essay. You can also enter the text manually.
            Check the corresponding checkboxes to enable the desired features, and click the 'Submit Essay' button to process your essay.

            For more information or support, you can find our details below.""")

        st.write(" ")
        st.write(" ")
        

            # developers = [
            #     {
            #         "name": "Faris",
            #         "image": "images/faris.jpg",
            #         "background": " "
            #     },
            #     {
            #         "name": "Rabiatul",
            #         "image": "images/rab.jpg",
            #         "background": " "
            #     },
            #     {
            #         "name": "Najla",
            #         "image": "images/najla.jpg",
            #         "background": " "
            #     },
            #     {
            #         "name": "Fatin",
            #         "image": "images/fatin.png",
            #         "background": " "
            #     }
            # ]

            

            # col1, col2, col3, col4 = st.columns(4)

            # for i, developer in enumerate(developers):
            #     if i % 4 == 0:
            #         col = col1
            #     elif i % 4 == 1:
            #         col = col2
            #     elif i % 4 == 2:
            #         col = col3
            #     else:
            #         col = col4

            #     with col:
            #         st.subheader(developer["name"])
            #         st.image(developer["image"], width=120)
            #         st.write(developer["background"])



    if __name__ == '__main__':
        main()

if selected == "Pic-To-Words":
    # Modify code for "Pic-To-Words"
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

        st.title("🖼️ IlmNexa Pic-To-Words ")
        st.write("Upload an image and we'll give you ideas on what to write on the subject!")

        default_text = " "

        # Selectbox
        options = ["Argumentative","Persuasive", "Descriptive", "News Report", "Tutorial"]
        selected_option = st.selectbox("Select an essay type", options, key="pic_to_words_selectbox")

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pic_to_words_image_upload")

        if uploaded_image is not None:
            st.image(uploaded_image)  # Remove the 'key' argument
            image = np.array(cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1))
            
            # Save the image
            cv2.imwrite("uploaded_image.jpg", image)
            
            caption = img_to_text("uploaded_image.jpg")
            st.write("Caption: ")
            st.header(caption)


            topic = caption
            prompt = caption
            temp = 0.9

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

if selected == "Essay Tool":
    # Modify code for "Essay Tool"
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
        st.title(":coffee: IlmNexa Essay Tool")
        # st.markdown("<h2 style='text-align: center;'>Your Essay Enhancement Companion 🚀</h2>", unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="essay_tool_pdf_upload")

        # Display the uploaded file
        if uploaded_file is not None:
            default_text = extract_text_from_pdf(uploaded_file)

            for text in default_text:
                split_message = re.split(r'\s+|[ ]\s*', text)
                new_text = ' '.join(split_message)
            default_text = new_text
            user_input = new_text
            st.write("Uploaded filename:", uploaded_file.name)

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="essay_tool_image_upload")

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

if selected == "Study Buddy":
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

        st.title("🤖 IlmNexa Study Buddy ☕️")
        st.write("Welcome to IlmNexa Bot! Your friendly essay enhancement companion.")

        default_text = " "
        with st.expander("Click to enter your essay"):
            # File uploader
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="study_buddy_pdf_upload")

            # Display the uploaded file
            if uploaded_file is not None:
                default_text = extract_text_from_pdf(uploaded_file)

                for text in default_text:
                    split_message = re.split(r'\s+|[ ]\s*', text)
                    new_text = ' '.join(split_message)
                default_text = new_text
                user_input = new_text
                st.write("Uploaded filename:", uploaded_file.name)
            
        
            uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="study_buddy_image_upload")

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
        selected_option = st.selectbox("Select an essay type", options, key="study_buddy_selectbox")
        
    

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


if selected == "Note Taking":
    import streamlit as st
    import os
    from streamlit_img_label import st_img_label
    from streamlit_img_label.manage import ImageManager, ImageDirManager

    def run(img_dir, labels):

        st.title("🖼️ IlmNexa Note Taking ")
        st.write("Prepare notes for your essays!")

        st.set_option("deprecation.showfileUploaderEncoding", False)
        idm = ImageDirManager(img_dir)

        if "files" not in st.session_state:
            st.session_state["files"] = idm.get_all_files()
            st.session_state["annotation_files"] = idm.get_exist_annotation_files()
            st.session_state["image_index"] = 0
        else:
            idm.set_all_files(st.session_state["files"])
            idm.set_annotation_files(st.session_state["annotation_files"])
        
        def refresh():
            st.session_state["files"] = idm.get_all_files()
            st.session_state["annotation_files"] = idm.get_exist_annotation_files()
            st.session_state["image_index"] = 0

        def next_image():
            image_index = st.session_state["image_index"]
            if image_index < len(st.session_state["files"]) - 1:
                st.session_state["image_index"] += 1
            else:
                st.warning('This is the last image.')

        def previous_image():
            image_index = st.session_state["image_index"]
            if image_index > 0:
                st.session_state["image_index"] -= 1
            else:
                st.warning('This is the first image.')

        def next_annotate_file():
            image_index = st.session_state["image_index"]
            next_image_index = idm.get_next_annotation_image(image_index)
            if next_image_index:
                st.session_state["image_index"] = idm.get_next_annotation_image(image_index)
            else:
                st.warning("All images are annotated.")
                next_image()

        def go_to_image():
            file_index = st.session_state["files"].index(st.session_state["file"])
            st.session_state["image_index"] = file_index

        # Sidebar: show status
        n_files = len(st.session_state["files"])
        n_annotate_files = len(st.session_state["annotation_files"])
        st.sidebar.write("Total files:", n_files)
        st.sidebar.write("Total annotate files:", n_annotate_files)
        st.sidebar.write("Remaining files:", n_files - n_annotate_files)

        st.sidebar.selectbox(
            "Files",
            st.session_state["files"],
            index=st.session_state["image_index"],
            on_change=go_to_image,
            key="file",
        )
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button(label="Previous image", on_click=previous_image)
        with col2:
            st.button(label="Next image", on_click=next_image)
        st.sidebar.button(label="Next need annotate", on_click=next_annotate_file)
        st.sidebar.button(label="Refresh", on_click=refresh)

        # Upload image
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            # Save the uploaded image to the image directory
            with open(os.path.join(img_dir, uploaded_image.name), "wb") as f:
                f.write(uploaded_image.read())
            # Refresh the file list
            refresh()

        # Main content: annotate images
        img_file_name = idm.get_image(st.session_state["image_index"])
        img_path = os.path.join(img_dir, img_file_name)
        im = ImageManager(img_path)
        img = im.get_img()
        resized_img = im.resizing_img()
        resized_rects = im.get_resized_rects()
        rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

        def annotate():
            im.save_annotation()
            image_annotate_file_name = img_file_name.split(".")[0] + ".xml"
            if image_annotate_file_name not in st.session_state["annotation_files"]:
                st.session_state["annotation_files"].append(image_annotate_file_name)
            # next_annotate_file()

        if rects:
            st.button(label="Save", on_click=annotate)
            preview_imgs = im.init_annotation(rects)

            for i, prev_img in enumerate(preview_imgs):
                prev_img[0].thumbnail((200, 200))
                col1, col2 = st.columns(2)
                with col1:
                    col1.image(prev_img[0])
                with col2:
                    default_index = 0  # Default index if prev_img[1] is not in labels
                    if prev_img[1] in labels:
                        default_index = labels.index(prev_img[1])

                    custom_label = st.text_input("Take notes:", prev_img[1], key=f"label_{i}")
                    im.set_annotation(i, custom_label)

    if __name__ == "__main__":
        # Selectbox

        custom_labels = ["", "Needs Elaboration", "Wrong Spelling", "Good Writing!"]
        run("img_dir", custom_labels)
