import streamlit as st  #to build interface
from transformers import pipeline #run models
from huggingface_hub import snapshot_download #install arabic modal localy
import pdfplumber #extract text from pdf
import re #clean text
import torch

st.set_page_config(layout="wide")  #for interface

def clean_arabic_text(text):#make result butter cause arabic model is sensitve
    text=re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s.,!?؛،\-]"," ",text)#delete these symbols
    text=re.sub(r"\s+"," ",text)  #make space orginazed and clean
    return text.strip()

local_dir=snapshot_download("abdalrahmanshahrour/arabartsummarization")
@st.cache_resource #make modle install only once

def get_arabic_summarizer():
    return pipeline(
        "text2text-generation",
        model=local_dir,
        tokenizer=local_dir,
    )

@st.cache_resource
def get_english_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

def chunk_text(text,max_chunk_words=400):#cause the models cant summary 5000 tokens
    words=text.split()
    chunks=[]
    for i in range(0,len(words),max_chunk_words):
        chunk=" ".join(words[i:i+max_chunk_words])
        chunks.append(chunk)
    return chunks

def text_summary(text,lang_code):
    if lang_code == "ar":
        text=clean_arabic_text(text)
        summarizer=get_arabic_summarizer()
        output_key ="generated_text"
    else:
        summarizer=get_english_summarizer()
        output_key ="summary_text"

    chunks=chunk_text(text,max_chunk_words=250)        
    summaries=[]
    for chunk in chunks:
        result=summarizer(
                chunk,
                max_length=300,
                min_length=120,
                do_sample=False
                )
        summaries.append(result[0][output_key])
    return " ".join(summaries)    
   
   
def extract_text_from_pdf(file_path):
    # Open the PDF file using pdfplumber
    text=""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text =page.extract_text()
            if page_text:
                text+=page_text+"\n"
    return text



#streamlit 'User interface'


if "step" not in st.session_state:
    st.session_state.step ="input"
    st.session_state.texts=[]
    st.session_state.current_text=""


def reset_to_input():
    st.session_state.step="input"
    st.session_state.current_text=""
    st.rerun()

if st.session_state.step =="input":
    choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])
    
    if choice == "Summarize Text":
        st.subheader("Summarize Text")
        input_text = st.text_area("Enter your text here",value="\n".join(st.session_state.texts))
        if st.button("Summarize"):
            st.session_state.texts.append(input_text)
            st.session_state.step ="result"
            st.session_state.current_text= input_text

    elif  choice == "Summarize Document":
        st.subheader("Summarize Document")
        input_file =st.file_uploader("Upload your document here",type=['pdf'])
        if input_file and st.button("Summarize"):
            with open("doc_file.pdf","wb") as f:
                f.write(input_file.getbuffer())
            with st.spinner("Extracting text & summarizing..."):
                extracted_text=extract_text_from_pdf("doc_file.pdf")    
                st.session_state.texts.append(extracted_text)
                st.session_state.step ="result"
                st.session_state.current_text = extracted_text




if st.session_state.step =="result":
    text =st.session_state.current_text

    try:
        lang_code ="ar" if re.search(r'[\u0600-\u06FF]',text) else "en"
    except:
        lang_code ="en"    
 
    summary=text_summary(text,lang_code)
    st.subheader("Summary Result")
    st.success(summary)

    if st.button("back to home"):
        reset_to_input()


