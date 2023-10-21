import streamlit as st

def main():
    
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
