import streamlit as st
import os
import logging
from typing import List, Dict, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import OpenAI
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import hashlib
from datetime import datetime
from dotenv import load_dotenv
# Configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot with necessary configurations."""
        self.setup_api_keys()
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.index_name = "rag-chatbot-index"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.setup_pinecone_index()
        
        # Initialize OpenAI
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def setup_api_keys(self):
        """Setup API keys from environment variables."""
        if not os.getenv("PINECONE_API_KEY"):
            st.error("Please set PINECONE_API_KEY in your environment variables")
            st.stop()
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please set OPENAI_API_KEY in your environment variables")
            st.stop()
    
    def setup_pinecone_index(self):
        """Setup Pinecone index for storing embeddings."""
        try:
            # Check if index exists
            if self.index_name not in [index.name for index in self.pc.list_indexes()]:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {str(e)}")
            st.error(f"Error connecting to Pinecone: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise e
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Extract page information if available
            page_match = re.search(r'--- Page (\d+) ---', chunk_text)
            page_num = page_match.group(1) if page_match else "Unknown"
            
            # Clean chunk text (remove page markers)
            clean_chunk = re.sub(r'--- Page \d+ ---', '', chunk_text).strip()
            
            if len(clean_chunk) > 50:  # Only keep substantial chunks
                chunk_id = hashlib.md5(clean_chunk.encode()).hexdigest()[:16]
                chunks.append({
                    'id': chunk_id,
                    'text': clean_chunk,
                    'page': page_num,
                    'chunk_index': len(chunks)
                })
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Create embeddings for text chunks."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks
    
    def store_in_pinecone(self, chunks: List[Dict], document_name: str):
        """Store chunks and embeddings in Pinecone."""
        vectors = []
        
        for chunk in chunks:
            vectors.append({
                'id': f"{document_name}_{chunk['id']}",
                'values': chunk['embedding'],
                'metadata': {
                    'text': chunk['text'],
                    'page': chunk['page'],
                    'chunk_index': chunk['chunk_index'],
                    'document_name': document_name,
                    'timestamp': datetime.now().isoformat()
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        
        logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        relevant_chunks = []
        for match in results['matches']:
            relevant_chunks.append({
                'text': match['metadata']['text'],
                'page': match['metadata']['page'],
                'score': match['score'],
                'document_name': match['metadata']['document_name']
            })
        
        return relevant_chunks
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using OpenAI GPT with retrieved chunks."""
        if not relevant_chunks:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"[Page {chunk['page']}]: {chunk['text']}" 
            for chunk in relevant_chunks
        ])
        
        # Create prompt
        prompt = f"""Based ONLY on the provided document context, answer the following question. If the answer cannot be found in the context, say so clearly.

Context from document:
{context}

Question: {query}

Instructions:
1. Answer only based on the provided context
2. Include specific page references for your sources
3. If information is not in the context, state that clearly
4. Be concise but comprehensive

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided document context. Always cite your sources with page numbers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add source citations
            sources = list(set([f"Page {chunk['page']}" for chunk in relevant_chunks]))
            citations = f"\n\n**Sources:** {', '.join(sources)}"
            
            return answer + citations
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, query: str) -> str:
        """Main method to answer a question using RAG pipeline."""
        # Check if query is in scope (basic guardrail)
        if len(query.strip()) < 3:
            return "Please provide a more specific question."
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            return "I can only answer questions based on the provided document. Your question doesn't seem to match any content in the uploaded document."
        
        # Generate answer
        answer = self.generate_answer(query, relevant_chunks)
        return answer
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}

def main():
    st.set_page_config(
        page_title="RAG Chatbot with Pinecone",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot with Pinecone")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a PDF document to create a knowledge base"
        )
        
        if uploaded_file is not None and not st.session_state.document_processed:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Initialize chatbot
                        st.session_state.chatbot = RAGChatbot()
                        
                        # Extract text from PDF
                        pdf_text = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)
                        
                        # Chunk text
                        chunks = st.session_state.chatbot.chunk_text(pdf_text)
                        
                        # Create embeddings
                        chunks_with_embeddings = st.session_state.chatbot.create_embeddings(chunks)
                        
                        # Store in Pinecone
                        document_name = uploaded_file.name.replace('.pdf', '')
                        st.session_state.chatbot.store_in_pinecone(chunks_with_embeddings, document_name)
                        
                        st.session_state.document_processed = True
                        st.success(f"✅ Document processed successfully!")
                        st.info(f"Created {len(chunks)} chunks from your document")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        
        # Show index statistics
        if st.session_state.chatbot:
            st.header("📊 Index Statistics")
            stats = st.session_state.chatbot.get_index_stats()
            if stats:
                st.metric("Total Vectors", stats.get('total_vectors', 0))
                st.metric("Dimension", stats.get('dimension', 0))
                st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.2%}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.document_processed and st.session_state.chatbot:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Bot:** {answer}")
                st.divider()
        
        # Question input
        user_question = st.text_input(
            "Ask a question about the document:",
            placeholder="e.g., What are the main topics discussed in the document?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask", type="primary")
        
        if ask_button and user_question:
            with st.spinner("Thinking..."):
                answer = st.session_state.chatbot.answer_question(user_question)
                st.session_state.chat_history.append((user_question, answer))
                st.rerun()
        
        # Sample questions
        if not st.session_state.chat_history:
            st.markdown("### 💡 Sample Questions")
            sample_questions = [
                "What are the main topics covered in this document?",
                "Can you summarize the key points?",
                "What conclusions are drawn in the document?",
                "Are there any specific recommendations mentioned?"
            ]
            
            for question in sample_questions:
                if st.button(question, key=f"sample_{hash(question)}"):
                    with st.spinner("Thinking..."):
                        answer = st.session_state.chatbot.answer_question(question)
                        st.session_state.chat_history.append((question, answer))
                        st.rerun()
    
    else:
        st.info("👆 Please upload a PDF document to get started!")
        
        # Display features
        st.markdown("### ✨ Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **📄 PDF Processing**: Automatic text extraction and chunking
            - **🔍 Smart Search**: Vector similarity search with Pinecone
            - **🤖 AI Answers**: GPT-powered responses with citations
            """)
        
        with col2:
            st.markdown("""
            - **📚 Source Citations**: Page-level source references
            - **🛡️ Guardrails**: Only answers from uploaded document
            - **📊 Statistics**: Real-time index monitoring
            """)

if __name__ == "__main__":
    main()