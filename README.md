📄 Chat with Multiple PDFs — RAG Chatbot

A Streamlit-based PDF question-answering application that uses Retrieval-Augmented Generation (RAG) to answer questions using content extracted from uploaded PDF documents. It uses Google Gemini for embeddings and chat responses, and ChromaDB to store and retrieve relevant text chunks.

✨ Features

- Multiple PDF uploads — upload more than one PDF in the sidebar.
- PDF text extraction — extract text page by page using pypdf.
- Text chunking — split document text into overlapping chunks for retrieval.
- Semantic search — retrieve the most relevant chunks from ChromaDB for a question.
- AI-generated answers — use Google Gemini to answer based on retrieved context.
- Source references — show the source PDF, page number, and retrieved text under each answer.
- Chat history — keep the conversation visible during the Streamlit session.
- Re-indexing detection — avoid rebuilding the vector store on every Streamlit rerun when the upload signature is unchanged.

🧰 Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini (langchain-google-genai)
- ChromaDB (langchain-chroma)
- PyPDF (pypdf)
- Recursive Character Text Splitter
- python-dotenv


🏗️ How It Works

1. Upload one or more PDF files.
2. The app extracts text from each PDF page and records the filename and page number as metadata.
3. The extracted pages are split into chunks of 1,000 characters with 200 characters of overlap.
4. Google embedding models convert the chunks into vectors.
5. ChromaDB stores the vectors.
6. When you ask a question, the app retrieves the four most relevant chunks.
7. The retrieved context, chat history, and question are sent to Gemini.
8. The answer and its retrieved source details are displayed in the chat.


PDF uploads
    ↓
Extract text page by page
    ↓
Split into overlapping chunks
    ↓
Create Google Gemini embeddings
    ↓
Store vectors in ChromaDB
    ↓
User asks a question
    ↓
Retrieve relevant chunks
    ↓
Gemini generates an answer
    ↓
Display answer and source details

🗺️ Possible Future Improvements

Every citation shows the correct file name and page, even when two files are searched together.

A sidebar dropdown lets the user search All documents or one chosen file.

Changing the dropdown does not re-upload or re-index anything.
