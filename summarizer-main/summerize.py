import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import re
import tempfile
import os

st.set_page_config(layout="wide")

@st.cache_resource
def get_english_summarizer():
    """Download the English summary model"""
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def get_arabic_summarizer():
    """Download the Arabic language summary model"""
    try:
        return pipeline("summarization", model="abdalrahmanshahrour/arabartsummarization")
    except:
        # If the model fails to load, we will try to load it manually.
        tokenizer = AutoTokenizer.from_pretrained("abdalrahmanshahrour/arabartsummarization")
        model = AutoModelForSeq2SeqLM.from_pretrained("abdalrahmanshahrour/arabartsummarization")
        return pipeline("summarization", model=model, tokenizer=tokenizer)

def detect_language(text):
    """Detecting the language of the text"""
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    total_chars = len(text.replace(" ", "").replace("\n", ""))
    
    if total_chars == 0:
        return "en"
    
    arabic_ratio = arabic_chars / total_chars
    return "ar" if arabic_ratio > 0.3 else "en"

def smart_paragraph_split(text, max_para_length=500):
    """Dividing the text into smart paragraphs while keeping the sentences complete"""
    # Clean the text first
    text = re.sub(r'\s+', ' ', text)
    
    # Determining the segmentation pattern based on language
    sentences = re.split(r'(?<=[.!؟])\s+', text)
    
    paragraphs = []
    current_para = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_para) + len(sentence) + 1 <= max_para_length:
            current_para += " " + sentence if current_para else sentence
        else:
            if current_para:
                paragraphs.append(current_para)
            current_para = sentence
    
    if current_para:
        paragraphs.append(current_para)
    
    return paragraphs

def remove_duplicate_sentences(text):
    """Removing duplicate sentences from the text"""
    sentences = text.split(" . ")
    unique_sentences = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))
    clean_text = " . ".join(unique_sentences) + " ."
    return clean_text

def summarize_text(text, language="en"):
    """Summarize the text based on language"""    
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if language == "ar":
        # Removing duplicate sentences from the text
        text = remove_duplicate_sentences(text)
        summarizer = get_arabic_summarizer()
        
        # Dividing the text into paragraphs
        paragraphs = smart_paragraph_split(text, max_para_length=400)
        
        summaries = []
        progress_bar = st.progress(0)
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                try:
                    result = summarizer(para, max_length=150, min_length=40, do_sample=False)
                    summaries.append(result[0]['summary_text'])
                except Exception as e:
                    st.warning(f"⚠️ Error in summarizing paragraph {i+1}: {str(e)}")
                    summaries.append(para[:200] + "...")
            
            progress_bar.progress((i + 1) / len(paragraphs))
        
        # Merge the final summary
        full_summary = " ".join(summaries)
        full_summary = re.sub(r'\s+', ' ', full_summary)
        full_summary = re.sub(r'\.{2,}', '.', full_summary)
        
        return full_summary
    
    else:  # English
        summarizer = get_english_summarizer()
        
        # Divide the text into parts
        words = text.split()
        chunks = []
        chunk_size = 500
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            result = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(result[0]['summary_text'])
            progress_bar.progress((i + 1) / len(chunks))
        
        return " ".join(summaries)

def extract_text_from_pdf(file):
    """Extracting text from a PDF file"""
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
    
    return text

def extract_text_from_txt(file):
    """Extracting text from a txt file"""
    try:
        text = file.getvalue().decode("utf-8")
        return text
    except:
        try:
            text = file.getvalue().decode("utf-16")
            return text
        except Exception as e:
            st.error(f"Error reading the text file: {str(e)}")
            return ""

# User interface
st.title("📄 Text and Document Summarization Application")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Supported Summarization Languages")
    st.markdown("- English 🇺🇸")
    st.markdown("- Arabic 🇸🇦")
    
    st.markdown("---")
    st.subheader("Features")
    st.markdown("""
    - Summarize texts directly
    - Summarize PDF files
    - Summarize text files (txt)
    - Automatic language detection
    - User-friendly interface
    """)

# Main Menu
choice = st.selectbox("Choose the summary type:", [
    "Summarize Direct Text",
    "Summarize Document (PDF/TXT)"
])

if choice == "Summarize Direct Text":
    st.subheader("📝 Direct Text Summary")
    
    # Manual language selection (optional)
    language_option = st.radio("Choose your language (or leave it on automatic):", 
                              ["Automatic", "English", "Arabic"])
    
    input_text = st.text_area("Enter the text you want to summarize:", height=300)
    
    if st.button("Start Summarizing", type="primary"):
        if input_text.strip():
            with st.spinner("The text is being analyzed and summarized..."):
                # Select the language
                if language_option == "Automatic":
                    detected_lang = detect_language(input_text)
                    lang_name = "Arabic" if detected_lang == "ar" else "English"
                elif language_option == "Arabic":
                    detected_lang = "ar"
                    lang_name = "Arabic"
                else:
                    detected_lang = "en"
                    lang_name = "English"
                
                st.info(f"📊 Detected Language: {lang_name}")
                
                # Creating display columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**📄 Original Text**")
                    with st.expander("View Full Text", expanded=True):
                        st.text_area("", input_text, height=400, disabled=True)
                    
                    # Text statistics
                    st.markdown("**📊 Text Statistics**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Number of Words", len(input_text.split()))
                    with stats_col2:
                        st.metric("Number of Characters", len(input_text))
                    with stats_col3:
                        st.metric("Number of Sentences", len(re.split(r'[.!؟]+', input_text)))
                
                with col2:
                    st.markdown("**📌 Result After Summarizing**")
                    summary = summarize_text(input_text, detected_lang)
                    
                    with st.expander("Show Summary", expanded=True):
                        st.text_area("", summary, height=400, disabled=True)
                    
                    # Summary Statistics
                    st.markdown("**📊 Summary Statistics**")
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.metric("Word Count", len(summary.split()))
                    with stats_col2:
                        if len(input_text.split()) > 0:
                            reduction = ((len(input_text.split()) - len(summary.split())) / len(input_text.split())) * 100
                            st.metric("Reduction Percentage", f"{reduction:.1f}%")
                        else:
                            st.metric("Reduction Percentage", "0%")
                    
                    # Button to copy the summary
                    if st.button("📋 Copy the Summary"):
                        st.session_state['summary'] = summary
                        st.success("The summary has been copied to the clipboard!")
        else:
            st.warning("⚠️ Please enter text to summarize.")

elif choice == "Summarize Document (PDF/TXT)":
    st.subheader("📁 Document Summary")
    
    uploaded_file = st.file_uploader("Choose a PDF file or a text file", type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        # View file information
        file_details = {
            "File Name": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.json(file_details)
        
        if st.button("Start Summarizing the Document", type="primary"):
            with st.spinner("The document is being processed..."):
                # Extracting text based on file type
                if uploaded_file.type == "application/pdf":
                    extracted_text = extract_text_from_pdf(uploaded_file)
                else:  # Text file
                    extracted_text = extract_text_from_txt(uploaded_file)
                
                if extracted_text.strip():
                    # Detect the language
                    detected_lang = detect_language(extracted_text)
                    lang_name = "Arabic" if detected_lang == "ar" else "English"
                    
                    # Creating display columns
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**📄 Extracted Text**")
                        with st.expander("Display Extracted Text", expanded=True):
                            st.text_area("", extracted_text, height=400, disabled=True)
                        
                        st.info(f"🔤 Detected Language: {lang_name}")
                        
                        # Text statistics
                        st.markdown("**📊 Text Statistics**")
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("Word Count", len(extracted_text.split()))
                        with col_stat2:
                            st.metric("Number of Characters", len(extracted_text))
                    
                    with col2:
                        st.markdown("**📌 Final Summary**")
                        summary = summarize_text(extracted_text, detected_lang)
                        
                        with st.expander("View Summary", expanded=True):
                            st.text_area("", summary, height=400, disabled=True)
                        
                        # Summary statistics
                        st.markdown("**📊 Summary Statistics**")
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("Word Count", len(summary.split()))
                        with col_stat2:
                            if len(extracted_text.split()) > 0:
                                reduction = ((len(extracted_text.split()) - len(summary.split())) / 
                                           len(extracted_text.split())) * 100
                                st.metric("Reduction Percentage", f"{reduction:.1f}%")
                        
                        # Export Options
                        st.markdown("**💾 Export Summary**")
                        summary_file = f"Summary_{uploaded_file.name.split('.')[0]}.txt"
                        st.download_button(
                            label="📥 Save Summary as Text File",
                            data=summary,
                            file_name=summary_file,
                            mime="text/plain"
                        )
                else:
                    st.error("❌ No text was extracted from the file. Please ensure the file contains readable text.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Developed using 🤗 Transformers and Streamlit</p>
    <p>Models used: 
    <br>• sshleifer/distilbart-cnn-12-6 (English)
    <br>• abdalrahmanshahrour/arabartsummarization (Arabic)
    </p>
</div>
""", unsafe_allow_html=True)