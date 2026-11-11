# 📄 Chat with Multiple PDFs — RAG Chatbot

A Streamlit-based PDF question-answering application powered by **Retrieval-Augmented Generation (RAG)**. Upload multiple PDF documents, ask questions, and receive AI-generated answers based on the document content.

The application uses Google Gemini for embeddings and chat responses, and ChromaDB for storing and retrieving relevant document chunks.

## ✨ Features

- **Multiple PDF uploads** — Upload multiple PDF files at once.
- **PDF text extraction** — Extract text page by page using PyPDF.
- **Text chunking** — Split document text into smaller overlapping chunks for better retrieval.
- **Semantic search** — Retrieve relevant document chunks using ChromaDB.
- **AI-generated answers** — Generate contextual answers using Google Gemini.
- **Source references** — Display the correct PDF filename, page number, and retrieved text.
- **Document filtering** — Search across All Documents or select a specific PDF using the sidebar dropdown.
- **Document-specific retrieval** — Retrieve information only from the selected PDF.
- **No unnecessary re-indexing** — Changing the document selection does not re-upload or re-index PDFs.
- **Chat history** — Maintain conversation history during the Streamlit session.
- **Upload change detection** — Rebuild the vector store when uploaded PDF files change.

## 🧰 Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini (`langchain-google-genai`)
- ChromaDB (`langchain-chroma`)
- PyPDF (`pypdf`)
- Recursive Character Text Splitter
- python-dotenv

## 🏗️ How It Works

1. Upload one or more PDF files through the Streamlit sidebar.
2. Extract text from each PDF page.
3. Store the filename and page number as document metadata.
4. Split extracted text into chunks of 1,000 characters with 200 characters of overlap.
5. Generate embeddings using Google Gemini.
6. Store document vectors in ChromaDB.
7. Select All Documents or an individual PDF from the sidebar.
8. Retrieve the four most relevant chunks based on the question and selected document.
9. Send the retrieved context and question to Gemini.
10. Display the generated answer with its source filename and page number.

## 🔍 Document Search Options

The application provides two search options:

- **All Documents:** Search across all uploaded PDFs.
- **Specific PDF:** Search only within the selected PDF.

Changing the dropdown selection does not rebuild the vector store or re-index the uploaded documents.

## 🔄 Application Workflow

```text
Upload Multiple PDFs
        ↓
Extract Text Page by Page
        ↓
Split Text into Chunks
        ↓
Generate Gemini Embeddings
        ↓
Store Vectors in ChromaDB
        ↓
Select All Documents or One PDF
        ↓
Retrieve Relevant Chunks
        ↓
Generate Answer Using Gemini
        ↓
Display Answer and Source References
```
