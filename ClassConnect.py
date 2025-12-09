import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PIL import Image
import io
import numpy as np
import fitz  # PyMuPDF
from transformers import CLIPProcessor, CLIPModel

# Set environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# OpenAI API Key
OPENAI_API_KEY = ""  # Replace with your API key

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Extract text from PDFs
def get_pdf_content(pdf_files):
    raw_text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

# Split text into chunks
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Get embeddings for text chunks
def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Start a conversation with the PDF
def start_conversation(vector_embeddings):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_embeddings.as_retriever(), memory=memory
    )

# Extract images and associated text from PDFs
def extract_images_and_text(pdf_files):
    images_data = []
    for pdf_file in pdf_files:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text")
                image_list = page.get_images(full=True)
                for img in image_list:
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        images_data.append({
                            "page": page_num + 1,
                            "image": img.convert("RGB"),
                            "text": text.strip()
                        })
    return images_data

# Find the best match for a query
def find_best_match(query, images_data):
    text_inputs = clip_processor(
        text=query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    text_embedding = clip_model.get_text_features(**text_inputs).detach().numpy()
    best_match, highest_similarity = None, -1

    for data in images_data:
        image_inputs = clip_processor(
            images=data["image"],
            return_tensors="pt",
            padding=True
        )
        image_embedding = clip_model.get_image_features(**image_inputs).detach().numpy()
        similarity = np.dot(text_embedding, image_embedding.T)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = data

    return best_match

# Main Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatbot with Image Finder", layout="wide")
    st.write("Welcome to the PDF Chatbot with Image Finder!")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "images_data" not in st.session_state:
        st.session_state.images_data = None

    query = st.text_input("Enter your query (e.g., 'plant cell image'):")  # Define query here

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload PDF")
        pdf_files = st.file_uploader("Upload your PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Process PDF"):
            if pdf_files:
                st.session_state.images_data = extract_images_and_text(pdf_files)
                text = get_pdf_content(pdf_files)
                chunks = get_chunks(text)
                embeddings = get_embeddings(chunks)
                st.session_state.conversation = start_conversation(embeddings)
                st.success("PDF processed successfully!")
            else:
                st.warning("Please upload a PDF.")

    # Query Processing
    if query:
        if "images_data" in st.session_state and st.session_state.images_data:
            # Check if the query is asking for an image
            if "image" in query.lower():
                match = find_best_match(query, st.session_state.images_data)
                if match:
                    st.image(match["image"], caption=f"Page {match['page']}", use_container_width=True)
                else:
                    st.write("No matching image found.")
            else:
                # Assume the query is asking for text
                if st.session_state.conversation:
                    response = st.session_state.conversation.invoke(query)
                    # Extract only the answer part from the response
                    answer = response.get("answer", "No answer found.")
                    st.write(f"Answer: {answer}")
                else:
                    st.write("Please upload a PDF to process.")
        else:
            st.write("Please upload a PDF to process.")
            
if __name__ == "__main__":
    main()
