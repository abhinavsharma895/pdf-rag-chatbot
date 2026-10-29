# 📄 PDF RAG Chatbot

An AI-powered PDF chatbot built with Python, Streamlit, LangChain, Google Gemini, and ChromaDB. Upload a PDF, ask questions about its content, and receive context-aware answers with source citations.

## ✨ Features

- **PDF Upload:** Upload PDF documents directly through the Streamlit interface.
- **Text Extraction:** Extract text from PDF pages using PyPDF.
- **Text Chunking:** Split document text into smaller, overlapping chunks.
- **Vector Embeddings:** Generate embeddings using Google's Gemini embedding model.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB.
- **Semantic Search:** Retrieve the most relevant document chunks for each question.
- **AI-Powered Answers:** Generate answers using Google Gemini.
- **Source Citations:** Display source references and page numbers alongside answers.
- **Conversation History:** Maintain chat history while working with the same PDF.
- **Answer Evaluation:** Includes a DeepEval evaluation script for testing answer quality.

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- ChromaDB
- PyPDF
- DeepEval

## 🔍 How It Works

1. **Upload:** The user uploads a PDF document.
2. **Extract:** The application extracts text from each page.
3. **Chunk:** The extracted text is split into overlapping chunks.
4. **Embed:** Google Gemini converts the chunks into vector embeddings.
5. **Store:** ChromaDB stores the embeddings for similarity search.
6. **Retrieve:** The application retrieves the four most relevant chunks for a question.
7. **Generate:** Gemini generates an answer using the retrieved context and conversation history.
8. **Cite:** The application displays source references and page numbers.
9. 

## 🚀 Future Improvements
The user can upload several PDFs at once.

Every citation shows the correct file name and page, even when two files are searched together.

A sidebar dropdown lets the user search All documents or one chosen file.

Changing the dropdown does not re-upload or re-index anything.

