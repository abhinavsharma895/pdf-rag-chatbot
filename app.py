import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

import warnings
warnings.filterwarnings("ignore")


# Models (the same ones we used in class)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,   # low temperature = fewer made-up answers
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# ===== FUNCTIONS: paste new functions below this line =====


def load_pdf(uploaded_file):
    """Read the uploaded PDF and return one Document per page."""
    reader = PdfReader(uploaded_file)

    pages = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name, "page": page_number},
            )
        )
    return pages


def split_pages(pages):
    """Cut the pages into small overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # about 1000 characters per chunk
        chunk_overlap=200,  # chunks share 200 characters so no sentence is cut in half
    )
    return splitter.split_documents(pages)


def create_vector_store(chunks):
    """Embed every chunk and store the vectors in Chroma."""
    vector_store = Chroma(
        # a fresh name per PDF, so chunks from an old PDF never mix with the new one
        collection_name=f"pdf_{uuid.uuid4().hex}",
        embedding_function=embeddings,
    )
    vector_store.add_documents(chunks)
    return vector_store


def retrieve_chunks(question, selected_file):
    """Retrieve chunks from all PDFs or only the selected PDF."""

    search_kwargs = {"k": 4}

    if selected_file != "All Documents":
        search_kwargs["filter"] = {
            "source": selected_file
        }

    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs=search_kwargs
    )

    return retriever.invoke(question)


SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using PDF documents.

Rules:
1. Use only the provided context.
2. Cite facts using [SOURCE 1], [SOURCE 2], etc.
3. Each source contains a filename and page number.
4. Never mix information from different PDFs.
5. If information is not available, say:
   "I could not find this information in the selected documents."

Context:
{context}
"""

def format_context(docs):

    parts = []

    for i, doc in enumerate(docs, start=1):

        filename = doc.metadata["source"]
        page = doc.metadata["page"] + 1

        parts.append(
            f"""
[SOURCE {i}]
Filename: {filename}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(parts)


def build_sources(docs):
    """Make a clean list of sources to show under the answer."""
    sources = []
    for i, doc in enumerate(docs, start=1):
        sources.append({
            "id": i,
            "file": doc.metadata["source"],
            "page": doc.metadata["page"] + 1,
            "text": doc.page_content,
        })
    return sources


def ask_model(question, docs, history):
    """Send the context, the chat history and the question to the model."""
    context = format_context(docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    messages.append(HumanMessage(content=question))

    response = model.invoke(messages)
    return response.content[0]['text']


# ===== UI: everything that appears on the page =====
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF, then ask questions about it. Every answer comes with citations.")

uploaded_files = st.sidebar.file_uploader("Upload a PDF",
type="pdf",
accept_multiple_files=True
)

if not uploaded_files:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# Streamlit re-runs this whole file on every click, so we only index
# the PDF when it is a NEW file. Everything else lives in st.session_state.
file_key = "-".join(
    f"{file.name}-{file.size}"
    for file in uploaded_files
)


if st.session_state.get("file_key") != file_key:
    with st.spinner("Reading and indexing your PDF..."):
        all_chunks=[]
        total_pages=0

        for uploaded_file in uploaded_files:
            pages=load_pdf(uploaded_file)
            total_pages+=len(pages)
            chunks=split_pages(pages)
            all_chunks.extend(chunks)

        if len(all_chunks) == 0:
            st.sidebar.error("No text found. Is this a scanned PDF?")
            st.stop()

        st.session_state.vector_store = create_vector_store(all_chunks)
        st.session_state.file_key = file_key
        st.session_state.num_pages = total_pages
        st.session_state.num_chunks = len(all_chunks)
        st.session_state.messages = []   # new PDF = new conversation

# Document selection dropdown
pdf_names = [file.name for file in uploaded_files]

selected_file = st.sidebar.selectbox(
    "📚 Select Document",
    ["All Documents"] + pdf_names
)


def show_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['id']}] {s['file']} · page {s['page']}**")
            st.caption(s["text"])


# Show the conversation so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

# Wait for a new question
question = st.chat_input("Ask a question about your PDF")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            docs = retrieve_chunks(question, selected_file)
            answer = ask_model(question, docs, st.session_state.messages)
            sources = build_sources(docs)
        st.markdown(answer)
        show_sources(sources)

    # Save both messages so they appear again after the next re-run
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )