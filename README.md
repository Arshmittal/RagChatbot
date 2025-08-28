 🤖 RAG Chatbot with Pinecone + Streamlit

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **Pinecone**, and **OpenAI GPT**.  
It lets you upload a **PDF document** and then ask questions — the bot answers **only** from the document’s content with proper page citations.

---

## ✨ Features

- **PDF Processing** – Extracts and chunks text into overlapping sections  
-  **Vector Search** – Uses Pinecone for fast semantic search  
-  **AI Answers** – OpenAI GPT generates responses with citations  
- **Source References** – Page-level citations in every answer  
-  **Guardrails** – Refuses to answer out-of-scope queries  
- **Index Statistics** – Real-time Pinecone index monitoring  




---

## 🔧 Setup & Run

### 1. Clone repo
```bash
git clone <your-repo-url>



2. Add environment variables

Create a .env file in the project root and add your API keys:

PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key

3. Run the app

streamlit run app.py

Then open the provided local URL (usually http://localhost:8501).
