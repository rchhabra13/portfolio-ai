

import streamlit as st
import ollama
import os
import json
import time
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting storage
_request_counts = {}
_rate_limit_lock = threading.Lock()


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Pinecone Configuration
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="portfolio-rishi", env="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", env="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", env="PINECONE_REGION")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="phi", env="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", env="OLLAMA_EMBEDDING_MODEL")
    
    # Application Configuration
    app_env: str = Field(default="development", env="APP_ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_query_length: int = Field(default=500, env="MAX_QUERY_LENGTH")
    embedding_cache_size: int = Field(default=1000, env="EMBEDDING_CACHE_SIZE")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    
    # Security Configuration
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    max_requests_per_minute: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    
    # Embedding Configuration
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    chunk_size: int = Field(default=800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    
    # Search Configuration
    default_top_k: int = Field(default=3, env="DEFAULT_TOP_K")
    relevance_threshold: float = Field(default=0.7, env="RELEVANCE_THRESHOLD")
    
    @field_validator("pinecone_api_key")
    @classmethod
    def validate_pinecone_api_key(cls, v):
        if not v or v == "your_pinecone_api_key_here":
            raise ValueError("PINECONE_API_KEY must be set to a valid API key")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("max_query_length")
    @classmethod
    def validate_max_query_length(cls, v):
        if v < 10 or v > 2000:
            raise ValueError("MAX_QUERY_LENGTH must be between 10 and 2000")
        return v
    
    @field_validator("relevance_threshold")
    @classmethod
    def validate_relevance_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("RELEVANCE_THRESHOLD must be between 0.0 and 1.0")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class SearchResult(BaseModel):
    """Structured search result with validation."""
    text: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    id: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Structured chat message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    search_results: Optional[List[SearchResult]] = None


class ConversationMemory(BaseModel):
    """Conversation memory for context retention."""
    messages: List[ChatMessage] = Field(default_factory=list)
    user_id: str = Field(default="default")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class QueryAnalytics(BaseModel):
    """Query analytics for improving search quality."""
    query: str
    timestamp: datetime = Field(default_factory=datetime.now)
    search_results_count: int
    avg_relevance_score: float
    response_generated: bool
    user_satisfaction: Optional[float] = None


class PortfolioRAGChatbot:
    """Enhanced Portfolio RAG Chatbot with security, performance, and architecture improvements."""
    
    def __init__(self):
        try:
            self.settings = Settings()
            self.pc: Optional[Pinecone] = None
            self.index = None
            self.portfolio_data = ""
            self.chunks: List[str] = []
            self._embedding_cache = {}
            self._connection_pool = ThreadPoolExecutor(max_workers=4)
            self._last_health_check = None
            self._health_check_interval = 300  # 5 minutes
            
            # Conversation memory and analytics
            self._conversation_memories: Dict[str, ConversationMemory] = {}
            self._query_analytics: List[QueryAnalytics] = []
            self._max_conversation_length = 10  # Keep last 10 messages
            self._max_conversation_memories = 100  # Max 100 user sessions
            self._max_analytics_entries = 1000  # Max 1000 analytics entries
            
            # Configure Ollama client
            ollama.base_url = self.settings.ollama_base_url
            
            # Initialize data loading
            self._load_portfolio_data()
            
            logger.info("PortfolioRAGChatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PortfolioRAGChatbot: {e}")
            raise
        
    def _load_portfolio_data(self) -> None:
        """Load portfolio data from external file with error handling."""
        try:
            data_file = "portfolio_data.txt"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.portfolio_data = f.read()
                logger.info(f"Loaded portfolio data from {data_file}")
            else:
                logger.warning(f"Portfolio data file {data_file} not found, using embedded data")
                self.portfolio_data = self._get_embedded_portfolio_data()
        except Exception as e:
            logger.error(f"Failed to load portfolio data: {e}")
            self.portfolio_data = self._get_embedded_portfolio_data()
    
    def _get_embedded_portfolio_data(self) -> str:
        """Fallback embedded portfolio data."""
        return """
        Rishi Chhabra - AI & Machine Learning Engineer based in New York, NY

        ABOUT:
        AI/ML-focused software engineer specializing in designing and deploying intelligent systems (LLMs, semantic search, vector databases) with cloud-native MLOps. Proven ability to build scalable, privacy-preserving solutions for web, mobile, and enterprise applications.

        EDUCATION:
        - Stevens Institute of Technology — M.S. in Machine Learning (Sep 2024 – Present)
          Focus: Deep learning, computer vision, applied AI.
        - Central University of Haryana — B.Tech in Computer Science (Sep 2020 – May 2024)
          Focus: Advanced algorithms, software engineering, scalable systems.

        EXPERIENCE:
        - Senior Flutter Developer at Incuwise (Feb 2024 – Aug 2024)
          * Developed and launched 4 cross-platform mobile applications using Flutter and Node.js backends
          * Led migration of server infrastructure from Linode to AWS, significantly improving scalability and reliability
          * Enhanced user engagement and transaction security by integrating Firebase Cloud Messaging, Google Maps APIs, Stripe payment processing, and OAuth authentication
          * Collaborated with QA and design teams in an Agile environment to deliver high-quality features on schedule

        FEATURED PROJECTS:
        1. End-to-End MLOps Pipeline - Fully automated ML workflow on AWS using Docker, Kubernetes, and Apache Airflow for CI/CD and orchestration with 60% reduction in deployment time.
        
        2. LLM Fine-Tuning Pipeline on AWS - AWS-based pipeline to fine-tune large language models on domain-specific data using SageMaker for distributed training on GPU instances with spot instances and MLflow tracking.
        
        3. Agentic RAG with Reasoning - RAG system combining retrieval with reasoning chains for improved factual accuracy using LangChain and Llama3.
        
        4. Multimodal AI Assistant - Chatbot processing text and images using multimodal models (CLIP, Vision-LLM) for scene descriptions and visual question answering.
        
        5. xAI Finance Agent - Autonomous agent leveraging financial datasets and LLM reasoning to provide market insights, portfolio optimization, and real-time risk assessment.
        
        6. Advanced AI Agents Framework - Framework for scalable multi-agent orchestration, supporting autonomous task planning, delegation, and tool integration using LangChain and CrewAI.

        TECHNICAL SKILLS:
        Programming Languages: Python, Java, C++, JavaScript, TypeScript, Dart, SQL, R
        ML/AI Frameworks: TensorFlow, PyTorch, scikit-learn, Keras, OpenCV, NLTK, Hugging Face Transformers
        Web & Mobile: Flutter, React.js, Node.js (Express), GraphQL, Flask, Django, RESTful APIs
        Cloud & DevOps: AWS (Lambda, EC2, S3, DynamoDB, Rekognition, Bedrock, SageMaker), Microsoft Azure (AI services), Docker, Kubernetes, Terraform, CI/CD (GitHub Actions, Jenkins), MLflow, Apache Airflow
        Databases: PostgreSQL, MongoDB, MySQL, Firebase, Vector DBs (Pinecone, Milvus, FAISS)

        CONTACT:
        - LinkedIn: https://www.linkedin.com/in/rchhabra1/
        - Email: Rishi.chhabra@outlook.com
        - GitHub: https://github.com/rchhabra13
        """
    
    def _validate_input(self, query: str) -> Tuple[bool, str]:
        """Validate and sanitize user input."""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        query = query.strip()
        
        # Check length
        if len(query) > self.settings.max_query_length:
            return False, f"Query too long. Maximum length is {self.settings.max_query_length} characters"
        
        # Basic sanitization
        query = re.sub(r'[<>"\']', '', query)
        
        # Check for potential injection attempts
        suspicious_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'on\w+\s*=',
            r'alert\s*\(',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains potentially malicious content"
        
        return True, query
    
    def _check_rate_limit(self, user_id: str = "default") -> bool:
        """Check if user has exceeded rate limit."""
        if not self.settings.enable_rate_limiting:
            return True
        
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=1)
        
        with _rate_limit_lock:
            if user_id not in _request_counts:
                _request_counts[user_id] = []
            
            # Clean old requests
            _request_counts[user_id] = [
                req_time for req_time in _request_counts[user_id] 
                if req_time > minute_ago
            ]
            
            # Check if under limit
            if len(_request_counts[user_id]) < self.settings.max_requests_per_minute:
                _request_counts[user_id].append(current_time)
                return True
            
            return False
    
    def _health_check(self) -> bool:
        """Perform health check on external services."""
        current_time = datetime.now()
        
        # Skip if checked recently
        if (self._last_health_check and 
            current_time - self._last_health_check < timedelta(seconds=self._health_check_interval)):
            return True
        
        try:
            # Test Ollama connection
            if not self._test_ollama_connection():
                return False
            
            # Test Pinecone connection
            if self.index and not self._test_pinecone_connection():
                return False
            
            self._last_health_check = current_time
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _test_ollama_connection(self) -> bool:
        """Test Ollama connection with timeout."""
        try:
            # Test with a simple request
            response = ollama.chat(
                model=self.settings.ollama_chat_model,
                messages=[{"role": "user", "content": "test"}],
                options={"timeout": 10}
            )
            return bool(response and response.get('message'))
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
    
    def _test_pinecone_connection(self) -> bool:
        """Test Pinecone connection."""
        try:
            if not self.index:
                return False
            stats = self.index.describe_index_stats()
            return stats is not None
        except Exception as e:
            logger.error(f"Pinecone connection test failed: {e}")
            return False
    
    def initialize_pinecone(self) -> bool:
        """Initialize Pinecone connection with proper error handling."""
        try:
            if not self.settings.pinecone_api_key:
                raise ValueError("Pinecone API key not configured")
            
            self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.settings.pinecone_index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.settings.pinecone_index_name}")
                self.pc.create_index(
                    name=self.settings.pinecone_index_name,
                    dimension=self.settings.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.settings.pinecone_cloud,
                        region=self.settings.pinecone_region
                    )
                )
                time.sleep(10)  # Wait for index to be ready
                
            self.index = self.pc.Index(self.settings.pinecone_index_name)
            logger.info("Pinecone initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return False
    
    @lru_cache(maxsize=1000)
    def get_embedding_cached(self, text: str) -> Tuple[float, ...]:
        """Get embedding with caching for performance."""
        try:
            # Skip empty or very short texts
            if not text or len(text.strip()) < 3:
                logger.warning(f"Skipping embedding for text too short: '{text[:50]}...'")
                return tuple()
            
            response = ollama.embeddings(
                model=self.settings.ollama_embedding_model,
                prompt=text
            )
            embedding = response.get('embedding', [])
            if not embedding:
                logger.warning(f"Empty embedding returned for text: '{text[:50]}...'")
                return tuple()
            
            # Validate embedding dimensions
            if len(embedding) != self.settings.embedding_dimension:
                logger.warning(f"Embedding dimension mismatch: expected {self.settings.embedding_dimension}, got {len(embedding)}")
                return tuple()
            
            return tuple(embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed for text '{text[:50]}...': {e}")
            return tuple()
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (wrapper for cached version)."""
        cached_result = self.get_embedding_cached(text)
        return list(cached_result) if cached_result else []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), self.settings.batch_size):
            batch = texts[i:i + self.settings.batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as executor:
                batch_embeddings = list(executor.map(self.get_embedding, batch))
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def chunk_text_semantic(self, text: str) -> List[str]:
        """Split text using semantic-aware chunking."""
        if not text.strip():
            return []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        documents = splitter.create_documents([text])
        return [doc.page_content for doc in documents]
    
    def ingest_portfolio_data(self) -> bool:
        """Ingest portfolio data with improved error handling and validation."""
        try:
            if not self.portfolio_data.strip():
                logger.error("No portfolio data to ingest")
                return False
            
            # Check if data already exists
            if self.index:
                stats = self.index.describe_index_stats()
                if stats.total_vector_count > 0:
                    logger.info("Portfolio data already exists in the index")
                    return True
            
            # Chunk the text using semantic splitting
            self.chunks = self.chunk_text_semantic(self.portfolio_data)
            logger.info(f"Created {len(self.chunks)} text chunks")
            
            if not self.chunks:
                logger.error("No chunks created from portfolio data")
                return False
            
            # Generate embeddings in batches
            embeddings = self.get_embeddings_batch(self.chunks)
            
            # Filter out failed embeddings
            valid_embeddings = []
            for i, (chunk, embedding) in enumerate(zip(self.chunks, embeddings)):
                if embedding and len(embedding) == self.settings.embedding_dimension:
                    valid_embeddings.append({
                        "id": f"chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}",
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "chunk_index": i,
                            "source": "portfolio",
                            "created_at": datetime.now().isoformat()
                        }
                    })
            
            if not valid_embeddings:
                logger.error("No valid embeddings generated")
                return False
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(valid_embeddings), batch_size):
                batch = valid_embeddings[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(valid_embeddings)-1)//batch_size + 1}")
            
            logger.info(f"Successfully ingested {len(valid_embeddings)} vectors into Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return False
    
    def _get_conversation_context(self, user_id: str, max_messages: int = 3) -> str:
        """Get recent conversation context for better responses."""
        if user_id not in self._conversation_memories:
            return ""
        
        memory = self._conversation_memories[user_id]
        recent_messages = memory.messages[-max_messages:] if len(memory.messages) > max_messages else memory.messages
        
        context_parts = []
        for msg in recent_messages:
            if msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            else:
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _add_to_conversation_memory(self, user_id: str, message: ChatMessage) -> None:
        """Add message to conversation memory with memory management."""
        # Clean up old conversation memories if we have too many
        if len(self._conversation_memories) >= self._max_conversation_memories:
            self._cleanup_old_conversations()
        
        if user_id not in self._conversation_memories:
            self._conversation_memories[user_id] = ConversationMemory(user_id=user_id)
        
        memory = self._conversation_memories[user_id]
        memory.messages.append(message)
        memory.last_updated = datetime.now()
        
        # Keep only recent messages
        if len(memory.messages) > self._max_conversation_length:
            memory.messages = memory.messages[-self._max_conversation_length:]
    
    def _cleanup_old_conversations(self) -> None:
        """Clean up old conversation memories to prevent memory leaks."""
        if len(self._conversation_memories) <= self._max_conversation_memories:
            return
        
        # Sort by last updated time and remove oldest
        sorted_memories = sorted(
            self._conversation_memories.items(),
            key=lambda x: x[1].last_updated
        )
        
        # Remove oldest 20% of memories
        to_remove = len(sorted_memories) // 5
        for user_id, _ in sorted_memories[:to_remove]:
            del self._conversation_memories[user_id]
        
        logger.info(f"Cleaned up {to_remove} old conversation memories")
    
    def _record_query_analytics(self, query: str, search_results: List[SearchResult], response_generated: bool) -> None:
        """Record query analytics for improving search quality."""
        avg_score = sum(r.score for r in search_results) / len(search_results) if search_results else 0.0
        
        analytics = QueryAnalytics(
            query=query,
            search_results_count=len(search_results),
            avg_relevance_score=avg_score,
            response_generated=response_generated
        )
        
        self._query_analytics.append(analytics)
        
        # Keep only recent analytics to prevent memory growth
        if len(self._query_analytics) > self._max_analytics_entries:
            self._query_analytics = self._query_analytics[-self._max_analytics_entries:]
    
    def _optimize_context(self, search_results: List[SearchResult], max_length: int = 2000) -> str:
        """Optimize context by deduplicating and managing length."""
        if not search_results:
            return ""
        
        # Deduplicate similar content
        seen_texts = set()
        unique_results = []
        
        for result in search_results:
            # Simple deduplication based on text similarity
            text_hash = hashlib.md5(result.text.lower().encode()).hexdigest()[:16]
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(result)
        
        # Build context with length management
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(unique_results, 1):
            result_text = f"Context {i}:\n{result.text}"
            if current_length + len(result_text) <= max_length:
                context_parts.append(result_text)
                current_length += len(result_text)
            else:
                # Add partial result if there's space
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Only if there's meaningful space
                    partial_text = result.text[:remaining_space-50] + "..."
                    context_parts.append(f"Context {i}:\n{partial_text}")
                break
        
        return "\n\n".join(context_parts)
    
    def _adaptive_relevance_threshold(self, query: str) -> float:
        """Adaptive relevance threshold based on query characteristics."""
        base_threshold = self.settings.relevance_threshold
        
        # Lower threshold for specific technical queries
        technical_terms = ["python", "tensorflow", "pytorch", "aws", "docker", "kubernetes", "mlops", "rag"]
        if any(term in query.lower() for term in technical_terms):
            return max(0.5, base_threshold - 0.1)
        
        # Higher threshold for general questions
        general_terms = ["who", "what", "when", "where", "why", "how", "tell me about"]
        if any(term in query.lower() for term in general_terms):
            return min(0.9, base_threshold + 0.1)
        
        return base_threshold

    def _fallback_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Fallback search when Pinecone is not available."""
        try:
            # Simple keyword-based search in portfolio data
            query_lower = query.lower()
            portfolio_lower = self.portfolio_data.lower()
            
            # Split portfolio into chunks
            chunks = self.chunk_text_semantic(self.portfolio_data)
            
            # Score chunks based on keyword matches
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                score = 0
                
                # Count keyword matches
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 3:  # Only count meaningful words
                        score += chunk_lower.count(word)
                
                # Normalize score
                if len(chunk) > 0:
                    score = score / len(chunk.split())  # Normalize by chunk length
                
                if score > 0:
                    scored_chunks.append({
                        'text': chunk,
                        'score': min(score, 1.0),  # Cap at 1.0
                        'id': f'fallback_chunk_{i}',
                        'metadata': {'source': 'fallback_search'}
                    })
            
            # Sort by score and return top results
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            # Convert to SearchResult objects
            results = []
            for item in scored_chunks[:top_k]:
                try:
                    search_result = SearchResult(
                        text=item['text'],
                        score=item['score'],
                        id=item['id'],
                        metadata=item['metadata']
                    )
                    results.append(search_result)
                except Exception as e:
                    logger.warning(f"Failed to create fallback SearchResult: {e}")
                    continue
            
            logger.info(f"Fallback search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    def search_with_fallback(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Enhanced search with fallback mechanisms and relevance filtering."""
        if top_k is None:
            top_k = self.settings.default_top_k
        
        try:
            # Check if Pinecone is initialized
            if not self.index:
                logger.warning("Pinecone index not initialized, using fallback search")
                return self._fallback_search(query, top_k)
            
            # Get query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return self._fallback_search(query, top_k)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more results for filtering
                include_metadata=True
            )
            
            # Use adaptive relevance threshold
            threshold = self._adaptive_relevance_threshold(query)
            
            # Filter by adaptive relevance threshold
            filtered_results = []
            for match in results.matches:
                if match.score >= threshold:
                    try:
                        search_result = SearchResult(
                            text=match.metadata.get("text", ""),
                            score=float(match.score),
                            id=match.id,
                            metadata=match.metadata
                        )
                        filtered_results.append(search_result)
                    except Exception as e:
                        logger.warning(f"Failed to create SearchResult: {e}")
                        continue
            
            # If no good results, try with lower threshold
            if not filtered_results and threshold > 0.3:
                logger.info("No results above threshold, trying with lower threshold")
                for match in results.matches:
                    if match.score >= 0.3:
                        try:
                            search_result = SearchResult(
                                text=match.metadata.get("text", ""),
                                score=float(match.score),
                                id=match.id,
                                metadata=match.metadata
                            )
                            filtered_results.append(search_result)
                        except Exception as e:
                            logger.warning(f"Failed to create SearchResult: {e}")
                            continue
            
            # Return top results
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def generate_response(self, query: str, context: str = "", conversation_context: str = "") -> str:
        """Generate response with improved error handling and context management."""
        try:
            system_prompt = """You are Rishi Chhabra's AI portfolio assistant. You help answer questions about Rishi's background, projects, skills, and experience.

Key information about Rishi:
- AI & Machine Learning Engineer based in New York, NY
- Currently pursuing M.S. in Machine Learning at Stevens Institute of Technology
- Previously worked as Senior Flutter Developer at Incuwise
- Specializes in AI/ML, computer vision, MLOps, RAG systems, and cloud development
- Has extensive experience with AWS, Python, Flutter, and various AI frameworks
- Created numerous AI agents, RAG systems, and MLOps pipelines

Always be helpful, professional, and accurate. If you don't know something specific, say so rather than making it up.
Base your responses on the provided context when available. Keep responses concise but informative."""

            # Build comprehensive context
            context_parts = []
            if conversation_context:
                context_parts.append(f"Previous conversation:\n{conversation_context}")
            if context:
                context_parts.append(f"Relevant portfolio information:\n{context}")
            
            if context_parts:
                full_context = "\n\n".join(context_parts)
                user_message = f"{full_context}\n\nCurrent question: {query}\n\nPlease provide a helpful response based on the context above."
            else:
                user_message = f"User question: {query}\n\nPlease provide information about Rishi Chhabra based on what you know."
            
            response = ollama.chat(
                model=self.settings.ollama_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                options={"temperature": 0.7, "top_p": 0.9}
            )
            
            return response.message.content if response and response.message else "I apologize, but I couldn't generate a response at this time."
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def chat(self, query: str, user_id: str = "default") -> Tuple[str, List[SearchResult]]:
        """Main chat function with comprehensive error handling and validation."""
        try:
            # Validate input
            is_valid, processed_query = self._validate_input(query)
            if not is_valid:
                return f"Error: {processed_query}", []
            
            # Check rate limit
            if not self._check_rate_limit(user_id):
                return "Rate limit exceeded. Please wait a moment before asking another question.", []
            
            # Health check
            if not self._health_check():
                return "Service temporarily unavailable. Please try again later.", []
            
            # Get conversation context
            conversation_context = self._get_conversation_context(user_id)
            
            # Search for relevant context
            search_results = self.search_with_fallback(processed_query)
            
            # Optimize context with deduplication and length management
            context = self._optimize_context(search_results)
            
            # Generate response with conversation context
            response = self.generate_response(processed_query, context, conversation_context)
            
            # Record analytics
            self._record_query_analytics(processed_query, search_results, bool(response))
            
            # Add to conversation memory
            user_message = ChatMessage(role="user", content=processed_query)
            assistant_message = ChatMessage(role="assistant", content=response, search_results=search_results)
            
            self._add_to_conversation_memory(user_id, user_message)
            self._add_to_conversation_memory(user_id, assistant_message)
            
            return response, search_results
            
        except Exception as e:
            logger.error(f"Chat function failed: {e}")
            return f"I apologize, but I encountered an error: {str(e)}", []


def main():
    """Main Streamlit application with enhanced UI and error handling."""
    st.set_page_config(
        page_title="Rishi Chhabra - AI Portfolio Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Rishi Chhabra's AI Portfolio Chatbot")
    st.caption("Ask me anything about Rishi's background, projects, skills, and experience!")
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = PortfolioRAGChatbot()
            st.session_state.initialized = False
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")
            st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Display current settings
        st.subheader("Current Settings")
        settings = st.session_state.chatbot.settings
        st.write(f"**Environment:** {settings.app_env}")
        st.write(f"**Log Level:** {settings.log_level}")
        st.write(f"**Rate Limiting:** {'Enabled' if settings.enable_rate_limiting else 'Disabled'}")
        
        # Connection status
        st.subheader("Connection Status")
        
        if st.button("🔍 Test Connections"):
            with st.spinner("Testing connections..."):
                ollama_ok = st.session_state.chatbot._test_ollama_connection()
                pinecone_ok = st.session_state.chatbot._test_pinecone_connection()
                
                if ollama_ok:
                    st.success("✅ Ollama is working!")
                else:
                    st.error("❌ Ollama connection failed")
                
                if pinecone_ok:
                    st.success("✅ Pinecone is working!")
                else:
                    st.warning("⚠️ Pinecone not initialized")
        
        # Initialize chatbot
        if st.button("🚀 Initialize Chatbot"):
            with st.spinner("Initializing..."):
                if st.session_state.chatbot.initialize_pinecone():
                    if st.session_state.chatbot.ingest_portfolio_data():
                        st.session_state.initialized = True
                        st.success("✅ Chatbot initialized successfully!")
                    else:
                        st.error("❌ Failed to ingest data")
                else:
                    st.error("❌ Failed to initialize Pinecone")
        
        # Quick questions
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "Tell me about Rishi's background",
            "What are Rishi's technical skills?",
            "Show me Rishi's AI projects",
            "What is Rishi's experience with MLOps?",
            "How can I contact Rishi?",
            "What RAG systems has Rishi built?",
            "Tell me about Rishi's computer vision projects"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                st.session_state.selected_question = question
        
        # Analytics display
        if st.session_state.initialized and st.session_state.chatbot._query_analytics:
            st.subheader("📊 Query Analytics")
            analytics = st.session_state.chatbot._query_analytics[-10:]  # Last 10 queries
            
            if analytics:
                avg_score = sum(a.avg_relevance_score for a in analytics) / len(analytics)
                total_queries = len(st.session_state.chatbot._query_analytics)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", total_queries)
                with col2:
                    st.metric("Avg Relevance", f"{avg_score:.2f}")
                
                # Show recent queries
                with st.expander("Recent Queries", expanded=False):
                    for i, qa in enumerate(analytics[-5:], 1):
                        st.write(f"{i}. {qa.query[:50]}... (Score: {qa.avg_relevance_score:.2f})")
    
    # Main chat interface
    if st.session_state.initialized:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "search_results" in message:
                    with st.expander("🔍 Search Results", expanded=False):
                        for i, result in enumerate(message["search_results"], 1):
                            st.write(f"**Result {i}** (Score: {result.score:.3f})")
                            st.write(result.text[:200] + "..." if len(result.text) > 200 else result.text)
                            st.divider()
        
        # Handle quick question selection
        if hasattr(st.session_state, 'selected_question'):
            query = st.session_state.selected_question
            delattr(st.session_state, 'selected_question')
        else:
            query = st.chat_input("Ask me anything about Rishi's portfolio...")
        
        if query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, search_results = st.session_state.chatbot.chat(query)
                    st.markdown(response)
                    
                    # Show search results
                    if search_results:
                        with st.expander("🔍 Search Results", expanded=False):
                            for i, result in enumerate(search_results, 1):
                                st.write(f"**Result {i}** (Score: {result.score:.3f})")
                                st.write(result.text[:200] + "..." if len(result.text) > 200 else result.text)
                                st.divider()
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "search_results": search_results
            })
    
    else:
        st.warning("⚠️ Please initialize the chatbot using the sidebar.")
        
        # Setup instructions
        with st.expander("📖 Setup Instructions", expanded=True):
            st.markdown("""
            ### Prerequisites:
            1. **Environment Variables**: Create a `.env` file with your configuration:
               ```
               PINECONE_API_KEY=your_actual_api_key
               OLLAMA_BASE_URL=http://localhost:11434
               ```
            
            2. **Ollama**: Make sure Ollama is running locally with these models:
               ```bash
               ollama pull phi
               ollama pull nomic-embed-text
               ollama serve
               ```
            
            ### Steps:
            1. Ensure Ollama is running: `ollama serve`
            2. Click "Test Connections" to verify services
            3. Click "Initialize Chatbot" to set up the system
            4. Start chatting!
            
            ### Features:
            - ✅ Secure API key management
            - ✅ Performance optimizations with caching
            - ✅ Enhanced error handling
            - ✅ Input validation and sanitization
            - ✅ Rate limiting
            - ✅ Health monitoring
            - ✅ Semantic text chunking
            - ✅ Relevance filtering
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, Ollama, and Pinecone | Enhanced with Security & Performance Improvements")


if __name__ == "__main__":
    main()