import streamlit as st
import os
from rag_system import StudyRAGSystem

# Initialize the RAG system
if 'rag' not in st.session_state:
    st.session_state.rag = StudyRAGSystem()

# App Interface
st.set_page_config(page_title="Study Assistant", layout="wide")
st.title("📚 AI Study Assistant")

# File Upload
with st.sidebar:
    st.header("Upload Materials")
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process"):
        with st.spinner("Indexing..."):
            file_paths = []
            os.makedirs("data", exist_ok=True)
            for file in uploaded_files:
                path = os.path.join("data", file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(path)
            st.session_state.rag.index_documents(file_paths)
            st.success(f"Processed {len(uploaded_files)} files!")

# Main Interaction
tab1, tab2 = st.tabs(["Study Help", "Faculty Tools"])

with tab1:
    st.header("Student Assistance")
    query = st.text_area("Ask about study materials")
    if st.button("Generate Notes"):
        with st.spinner("Creating notes..."):
            response = st.session_state.rag.generate_student_response(query)
            st.markdown(response)

with tab2:
    st.header("Assessment Generator")
    topic = st.text_input("Topic for assessment")
    if st.button("Create Quiz"):
        with st.spinner("Generating questions..."):
            response = st.session_state.rag.generate_faculty_response(topic)
            st.markdown(response)