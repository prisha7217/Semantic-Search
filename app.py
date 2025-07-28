import streamlit as st
import pdfplumber
import docx
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import torch
import os
import re
import pickle
import nltk
nltk.download('punkt_tab')


st.title("Semantic Search")
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
paragraphs = []
sentences = []
sources = []
paragraph_embeddings = None
sentence_embeddings = None
processed_files = set()

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_corpus():
    global paragraphs, paragraph_embeddings
    global sentences, sentence_embeddings
    global sources, processed_files

    if os.path.exists("corpus_data.pkl"):
        with open("corpus_data.pkl", "rb") as f:
            data = pickle.load(f)
            paragraphs = data["paragraphs"]
            paragraph_embeddings = data["paragraph_embeddings"]
            sentences = data["sentences"]
            sentence_embeddings = data["sentence_embeddings"]
            sources = data["sources"]
            processed_files = data["processed_files"]
    else:
        paragraphs, paragraph_embeddings = [], None
        sentences, sentence_embeddings = [], None
        sources, processed_files = [], set()

load_corpus()

def extract_text_from_file(path):
    if path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            return "\n".join([page.extract_text() or '' for page in pdf.pages])
    elif path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif path.endswith(".txt"):
        with open(path, "r", encoding='utf-8') as f:
            return f.read()
    return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text(text, mode='paragraph'):
    if mode == 'sentence':
        return sent_tokenize(text)
    elif mode == 'paragraph':
        sentences = sent_tokenize(text)
        paragraphs = []
        chunk = []
        for s in sentences:
            chunk.append(s)
            if len(chunk) >= 3:
                paragraphs.append(' '.join(chunk))
                chunk = []
        if chunk:
            paragraphs.append(' '.join(chunk))
        return paragraphs
    else:
        raise ValueError("Mode must be 'sentence' or 'paragraph'")
    
def refresh_corpus(upload_dir):
    global paragraph_embeddings, sentence_embeddings
    global paragraphs, sentences, sources, processed_files

    new_paragraphs = []
    new_sentences = []
    new_sources_paragraphs = []
    new_sources_sentences = []

    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        if filename in processed_files:
            continue

        text = extract_text_from_file(filepath)

        para_chunks = split_text(text, mode='paragraph')
        sent_chunks = split_text(text, mode='sentence')

        new_paragraphs.extend(para_chunks)
        new_sources_paragraphs.extend([filename] * len(para_chunks))

        new_sentences.extend(sent_chunks)
        new_sources_sentences.extend([filename] * len(sent_chunks))

        processed_files.add(filename)

    if not new_paragraphs and not new_sentences:
        st.info("No new files to process.")
        return

    if new_paragraphs:
        new_para_embeddings = model.encode(new_paragraphs, convert_to_tensor=True, show_progress_bar=True)
        paragraphs.extend(new_paragraphs)
        sources.extend(new_sources_paragraphs)
        if paragraph_embeddings is None:
            paragraph_embeddings = new_para_embeddings
        else:
            paragraph_embeddings = torch.cat([paragraph_embeddings, new_para_embeddings], dim=0)

    if new_sentences:
        new_sent_embeddings = model.encode(new_sentences, convert_to_tensor=True, show_progress_bar=True)
        sentences.extend(new_sentences)
        if sentence_embeddings is None:
            sentence_embeddings = new_sent_embeddings
        else:
            sentence_embeddings = torch.cat([sentence_embeddings, new_sent_embeddings], dim=0)

    st.success(f"Processed {len(processed_files)} file(s) successfully.")

def save_corpus():
    with open("corpus_data.pkl", "wb") as f:
        pickle.dump({
            "paragraphs": paragraphs,
            "paragraph_embeddings": paragraph_embeddings,
            "sentences": sentences,
            "sentence_embeddings": sentence_embeddings,
            "sources": sources,
            "processed_files": processed_files
        }, f)

def delete_files(files_to_delete, upload_dir="uploaded_docs"):
    global paragraph_embeddings, sentence_embeddings
    global paragraphs, sentences, sources, processed_files

    if not files_to_delete:
        st.warning("No files selected to delete.")
        return

    for filename in files_to_delete:
        try:
            os.remove(os.path.join(upload_dir, filename))
            processed_files.discard(filename)
            if filename in st.session_state.get("uploaded_names", []):
                st.session_state.uploaded_names.remove(filename)
        except FileNotFoundError:
            pass

    retained_indices = [i for i, fname in enumerate(sources) if fname not in files_to_delete]

    paragraphs = [paragraphs[i] for i in retained_indices]
    sentences = [sentences[i] for i in retained_indices]
    sources = [sources[i] for i in retained_indices]

    if paragraph_embeddings is not None and retained_indices:
        paragraph_embeddings = paragraph_embeddings[retained_indices]
    else:
        paragraph_embeddings = None

    if sentence_embeddings is not None and retained_indices:
        sentence_embeddings = sentence_embeddings[retained_indices]
    else:
        sentence_embeddings = None

    save_corpus()
    st.success(f"Deleted {len(files_to_delete)} file(s) and cleaned up related memory.")


def semantic_search(query, k=5, mode="paragraph"):
    global paragraph_embeddings, sentence_embeddings, paragraphs, sentences, sources
    if mode == 'paragraph':
        if paragraph_embeddings is None or len(paragraphs) == 0:
            st.error("Paragraph corpus is empty. Please refresh the corpus first.")
            return []
        embeddings = paragraph_embeddings
        content = paragraphs
        label = "Paragraph"
        filenames = sources 
    else:
        if sentence_embeddings is None or len(sentences) == 0:
            st.error("Sentence corpus is empty. Please refresh the corpus first.")
            return []
        embeddings = sentence_embeddings
        content = sentences
        label = "Sentence"
        filenames = sources
    
    query_embedding = model.encode(query, convert_to_tensor=True)

    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    top_indices = torch.topk(cosine_scores, k=k).indices

    results = []
    for idx in top_indices:
        score = cosine_scores[idx].item()
        snippet = content[idx]
        source = filenames[idx] if filenames else "unknown"
        results.append((score, snippet, source))

    print(f"\nTop {k} matches ({label}-level):\n")
    for score, snippet, source in results:
        print(f"({score:.4f}) [{source}]\n{snippet}\n")

    return results


st.subheader("Upload Files")
uploaded_files = st.file_uploader("Upload files", type=["txt", "pdf", "docx"], accept_multiple_files=True, key="file_uploader")
if uploaded_files:
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []

    new_files_uploaded = False

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_names:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.uploaded_names.append(uploaded_file.name)
            new_files_uploaded = True

    if new_files_uploaded:
        st.success("New file(s) uploaded.")
        refresh_corpus(UPLOAD_DIR)
        save_corpus()
        
        if "file_uploader" in st.session_state:
            del st.session_state["file_uploader"]
        st.rerun()
    else:
        st.info("No new files to upload.")

uploaded_files_list = os.listdir(UPLOAD_DIR)
st.subheader("Uploaded Files")
st.markdown("#### Currently Uploaded:")
for fname in uploaded_files_list:
    st.markdown(f"- {fname}")

st.subheader("Delete Uploaded Files")

uploaded_files_list = os.listdir(UPLOAD_DIR)
if uploaded_files_list:
    selected_to_delete = st.multiselect("Select file(s) to delete", uploaded_files_list)
    if st.button("Delete Files"):
        delete_files(selected_to_delete, UPLOAD_DIR)
        st.success(f"Deleted: {selected_to_delete}")
        if "file_uploader" in st.session_state:
            del st.session_state["file_uploader"]
        st.rerun() 
else:
    st.info("No uploaded files found.")

st.subheader("Semantic Search")
query = st.text_input("Enter your query")

mode = st.selectbox("Search Mode", ["paragraph", "sentence"])

if query and st.button("Search"):
    results = semantic_search(query, mode=mode)
    st.markdown(f"### Top Matches ({mode.title()}-level)")
    for score, snippet, source in results:
        st.markdown(f"**({score:.4f}) [{source}]**")
        st.markdown(f"> {snippet}")
