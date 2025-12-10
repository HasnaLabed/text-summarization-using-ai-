# Abstractive Text Summarization Web-App

This project is a Streamlit-based web application that performs abstractive text summarization for both Arabic and English texts.
The app detects the language automatically, splits long documents into chunks, summarizes each chunk using transformer models, and returns a smooth, readable summary. 



NOTE:
The summarization models used in this project are not developed by me.They are publicly available pretrained models from HuggingFace.
<a href="#Documentations">Read More</a>

<br>


My Streamlit app allows us to process both raw text and PDF files to get a summary.
## Features
- Summarize Arabic and English text.
- Automatic language detection.
- Support PDF file uploads.
- Handles lomg texts bu chunking.
- Abstractive summarization (creates new text rather than copying)

## Resources
- <a href="https://summarizze.streamlit.app/">Click for Live Demo</a>
<!-- - <a href="https://www.canva.com/design/DAFiomy01y0/c-0xFFUA2sYer-fgyocu9g/view">Click for Presentation</a>-->
<!-- - <a href="https://docs.google.com/document/d/e/2PACX-1vQTKY3eI-kxC6N_Qj9QNt9AmdMPflHL3Qa8MvX75166BxBEKX-Muz3liu6_z0BBhrGJsl_ysDUY0gm2/pub">Click for Report</a> -->
- <a href="#Documentations">Click for Documentation</a>


# Pre-requisites
* [x] Any IDE
* [x] transformers `pip install transformers`
* [x] streamlit `pip install streamlit`
* [x] pdfplumber `pip install pdfplumber`
* [x] huggingface_hub `pip install huggingface_hub`
* [x] torch `pip install torch`

# Run the App
- Create virtual enviroment `-m venv venv       
  venv\Scripts\ activate` 
- Install the dependencies `pip install -r requuirement.txt`
- Execute `streamlit run app.py`

# How it works
1-Text Input
You can:
Enter text manually, or Upload a PDF file
PDF text is extracted using pdfplumber.

2-Language Detection
the app checks for arabic Unicode characters "lang_code = "ar" if re.search(r'[\u0600-\u06FF]', text) else "en""

3-Chuncking Long Text
Because transformer models cannot summarize very long sequences, the text is split:"chunk_text(text, max_chunk_words=250)"
Each chunk is summarized separately.

4-summarization Models
-Arabic summarizer
Uses HuggingFace pipeline "pipeline("text2text-generation", model="arabartsummarization", tokenizer="arabartsummarization")"
-English summarizer
"pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")"

5-Output
All summaries are combined into a clean, readable final summary.

# Model Used
###Arabic Model
-Name: abdalrahmanshahrour/arabartsummarization
-Pipeline: "text2text-generation"
-Type :Abstractive summarization ,based on T5-like transformer

###English Model
-Name: sshleifer/distilbart-cnn-12-6
-Pipeline: "summarization"
-Type: Distilled BART

**Both models are pretrained and not created by me

# Documentation
-[Abstractive Text Summarization (Research Paper)](https://www.researchgate.net/profile/N-Moratanch/publication/305912913_A_survey_on_abstractive_text_summarization)

-[HuggingFace Transformers](https://huggingface.co/docs/transformers)

-[Streamlit Documentation](https://docs.streamlit.io/)

-[pdfplumber](https://github.com/jsvine/pdfplumber)

