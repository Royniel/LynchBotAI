"""
Lynch RAG Pipeline v2.2 + Validation - Production Ready
A robust RAG system for Peter Lynch investment Q&A using only curated dataset content
Enhanced with improved cleaning, better error handling, and smarter text processing.
Now uses a simpler, conversational prompt and post-generation validation to remove links
and fix a few known bad patterns.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Try to import NLTK for better sentence splitting
try:
    import nltk
    nltk.download('punkt', quiet=True)
    USE_NLTK = True
except ImportError:
    USE_NLTK = False
    print("Warning: NLTK not installed. Using regex for sentence splitting (less accurate).")

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RAGConfig:
    """Central configuration for the RAG pipeline"""
    # Retrieval settings
    min_similarity_score: float = 0.6
    top_k_retrieval: int = 5
    enable_semantic_boost: bool = True
    boost_factor: float = 0.05
    
    # Generation settings  
    max_length: int = 400
    min_length: int = 80
    num_beams: int = 8
    no_repeat_ngram_size: int = 4
    repetition_penalty: float = 2.3
    length_penalty: float = 1.2
    temperature: float = 0.7
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model: str = "google/flan-t5-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cache settings
    enable_cache: bool = True
    max_cache_size: int = 100
    cache_ttl_seconds: int = 3600
    
    # Answer cleaning settings
    answer_min_length: int = 30
    answer_max_length: int = 800
    
    # Logging
    log_level: str = "INFO"
    log_queries: bool = True

    # Conversation settings
    # For FLAN-T5-base we disable chat history in the prompt to avoid it copying old turns.
    use_chat_history_in_prompt: bool = False

# Initialize config
config = RAGConfig()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# RATIO DESCRIPTIONS
# ============================================================

try:
    from ratio_explanations import RATIO_DESCRIPTIONS
except ImportError:
    RATIO_DESCRIPTIONS = {
        "pe_ratio": "Price-to-Earnings ratio - stock price relative to earnings per share. Lynch preferred P/E under 15 for value plays.",
        "peg_ratio": "Price/Earnings to Growth ratio - P/E divided by growth rate. Lynch's favorite metric; under 1.0 is attractive.",
        "pb_ratio": "Price-to-Book ratio - market value vs book value. Under 1.0 might indicate undervaluation.",
        "ps_ratio": "Price-to-Sales ratio - market cap divided by revenue. Lower is generally better.",
        "debt_to_equity": "Debt-to-Equity ratio - total debt vs shareholder equity. Lynch preferred low debt companies.",
        "current_ratio": "Current assets divided by current liabilities. Above 1.5 indicates good liquidity.",
        "roe": "Return on Equity - net income vs shareholder equity. Higher indicates efficiency.",
        "profit_margin": "Net income as percentage of revenue. Higher margins show pricing power.",
        "earnings_growth": "Year-over-year earnings growth rate. Lynch looked for consistent 15-20% growth.",
        "dividend_yield": "Annual dividends per share divided by stock price. Lynch liked growing dividends."
    }

# Lynch-specific terms for semantic boosting
LYNCH_KEYWORDS = {
    "peg": 0.08,
    "ten-bagger": 0.10,
    "ten bagger": 0.10,
    "multibagger": 0.08,
    "earnings growth": 0.06,
    "p/e ratio": 0.05,
    "price-to-earnings": 0.05,
    "fundamental analysis": 0.05,
    "growth stock": 0.05,
    "value stock": 0.05,
    "turnaround": 0.05,
    "stalwart": 0.05,
    "fast grower": 0.06,
    "slow grower": 0.04,
    "cyclical": 0.04,
    "magellan fund": 0.06,
    "invest in what you know": 0.07,
}

# ============================================================
# CACHE MANAGER
# ============================================================

class CacheManager:
    """Manages query result caching for performance"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.enabled = config.enable_cache
        
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters"""
        cache_data = {"query": query.lower().strip(), **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Retrieve from cache if available and not expired"""
        if not self.enabled:
            return None
            
        key = self._get_cache_key(query, **kwargs)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.config.cache_ttl_seconds:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, result: Any, **kwargs):
        """Store result in cache"""
        if not self.enabled:
            return
        
        # Manage cache size
        if len(self.cache) >= self.config.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = self._get_cache_key(query, **kwargs)
        self.cache[key] = (result, datetime.now().timestamp())
        logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Cache cleared")

# ============================================================
# CONVERSATION MANAGER
# ============================================================

class ConversationManager:
    """Manages multi-turn conversation state"""
    
    def __init__(self, max_history_tokens: int = 1000):
        self.sessions: Dict[str, List[Dict]] = {}
        self.max_history_tokens = max_history_tokens
        
    def add_turn(self, session_id: str, role: str, content: str):
        """Add a conversation turn"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim old history if too long
        self._trim_history(session_id)
    
    def _trim_history(self, session_id: str):
        """Keep conversation history within token limits"""
        history = self.sessions[session_id]
        
        # Simple token estimation (4 chars ≈ 1 token)
        total_chars = sum(len(turn["content"]) for turn in history)
        estimated_tokens = total_chars // 4
        
        while estimated_tokens > self.max_history_tokens and len(history) > 2:
            history.pop(0)
            total_chars = sum(len(turn["content"]) for turn in history)
            estimated_tokens = total_chars // 4
    
    def get_history(self, session_id: str, last_n: int = 8) -> List[Dict]:
        """Get recent conversation history"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id][-last_n:]
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def format_history(self, session_id: str) -> str:
        """Format history as a string for prompt"""
        history = self.get_history(session_id)
        if not history:
            return ""
        
        lines = []
        for turn in history:
            role = "LynchBot" if turn["role"] == "assistant" else "User"
            lines.append(f"{role}: {turn['content']}")
        
        return "Previous conversation:\n" + "\n".join(lines) + "\n\n"

# Normalizing the word lynch
def normalize_query_spelling(query: str) -> str:
    """
    Fix simple spelling variations for important Lynch-related terms.
    This keeps behavior predictable without adding heavy fuzzy-matching.
    """
    if not query:
        return query

    q = query.lower()

    CORRECTIONS = {
        "lych": "lynch",
        "lynchh": "lynch",
        "ly nch": "lynch",
        "lnych": "lynch",
        "lymch": "lynch",
        "lynh": "lynch",
        "lync": "lynch",        # common typo
        "pegg": "peg",
        "ppg ratio": "peg ratio",
    }

    for wrong, correct in CORRECTIONS.items():
        if wrong in q:
            q = q.replace(wrong, correct)

    return q


# ============================================================
# DATA LOADING & VALIDATION
# ============================================================

def load_and_validate_dataset(data_path: str) -> pd.DataFrame:
    """Load and validate the Lynch dataset"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Lynch dataset not found at: {data_path}")
    
    logger.info(f"Loading Lynch dataset from: {data_path}")
    df = pd.read_excel(data_path)
    
    required_cols = {"Questions", "Answers"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    
    # Clean data
    df["Questions"] = df["Questions"].astype(str).str.strip()
    df["Answers"] = df["Answers"].astype(str).str.strip()
    
    # Remove empty entries
    initial_count = len(df)
    df = df[(df["Questions"] != "") & (df["Answers"] != "")]
    removed = initial_count - len(df)
    
    if removed > 0:
        logger.warning(f"Removed {removed} empty Q&A pairs")
    
    logger.info(f"✅ Loaded {len(df)} valid Q&A entries")
    return df

# ============================================================
# MODELS INITIALIZATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "lynch_dataset.xlsx")

# Load dataset
df = load_and_validate_dataset(DATA_PATH)

# Create enhanced corpus with labels
corpus_texts: List[str] = []
corpus_metadata: List[Dict] = []

for idx, row in df.iterrows():
    q = row["Questions"]
    a = row["Answers"]
    label = row.get("Label", "")
    
    # Enhanced corpus entry
    corpus_entry = f"Question: {q}\nAnswer: {a}"
    if label:
        corpus_entry = f"Category: {label}\n{corpus_entry}"
    
    corpus_texts.append(corpus_entry)
    corpus_metadata.append({
        "index": idx,
        "question": q,
        "answer": a,
        "label": label
    })

logger.info("🧠 Loading sentence embedding model...")
embed_model = SentenceTransformer(config.embedding_model)
if config.device == "cuda":
    embed_model = embed_model.cuda()

logger.info("🧠 Loading FLAN-T5 generator...")
gen_tokenizer = T5Tokenizer.from_pretrained(config.generator_model)
gen_model = T5ForConditionalGeneration.from_pretrained(config.generator_model)
gen_model = gen_model.to(config.device)
gen_model.eval()

logger.info("📌 Computing corpus embeddings...")
corpus_embeddings = embed_model.encode(
    corpus_texts,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=config.device
)
logger.info("✅ Models and embeddings ready")

# Initialize managers
cache_manager = CacheManager(config)
conversation_manager = ConversationManager()

# ============================================================
# ENHANCED RETRIEVAL
# ============================================================

def _calculate_semantic_boost(query: str, answer: str) -> float:
    """Calculate boost score based on Lynch-specific keywords"""
    if not config.enable_semantic_boost:
        return 0.0
    
    boost = 0.0
    query_lower = query.lower()
    answer_lower = answer.lower()
    
    for term, weight in LYNCH_KEYWORDS.items():
        if term in query_lower and term in answer_lower:
            boost += weight
    
    return min(boost, 0.3)  # Cap maximum boost

def _retrieve_relevant_qa(
    query: str,
    top_k: int = None,
    min_score: float = None,
    use_cache: bool = True
) -> List[Dict]:
    """
    Enhanced retrieval with semantic boosting and caching
    Returns: List of result dictionaries
    
    Note: For internal use, call _retrieve_relevant_qa_with_cache_info 
    to get (results, cache_hit) tuple
    """
    results, _ = _retrieve_relevant_qa_with_cache_info(
        query, top_k, min_score, use_cache
    )
    return results

def _retrieve_relevant_qa_with_cache_info(
    query: str,
    top_k: int = None,
    min_score: float = None,
    use_cache: bool = True
) -> Tuple[List[Dict], bool]:
    """
    Internal retrieval function that returns cache hit info
    Returns: (results, cache_hit)
    """
    query = (query or "").strip()
    if not query:
        return [], False
    
    top_k = top_k or config.top_k_retrieval
    min_score = min_score or config.min_similarity_score
    
    # Check cache
    cache_hit = False
    if use_cache:
        cached = cache_manager.get(query, top_k=top_k, min_score=min_score)
        if cached is not None:
            return cached, True
    
    # Compute query embedding
    query_emb = embed_model.encode(query, convert_to_tensor=True, device=config.device)
    cos_scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    
    # Get top results
    top_k = min(top_k, len(corpus_texts))
    top_results = torch.topk(cos_scores, k=top_k)
    
    # Apply threshold
    if float(top_results.values[0]) < min_score:
        logger.debug(f"Best score {top_results.values[0]:.3f} below threshold {min_score}")
        return [], False
    
    # Build results with metadata and boosting
    results: List[Dict] = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        meta = corpus_metadata[idx]
        
        # Calculate semantic boost
        boost = _calculate_semantic_boost(query, meta["answer"])
        adjusted_score = float(score) + boost
        
        results.append({
            "score": float(score),
            "adjusted_score": adjusted_score,
            "boost": boost,
            "qa_text": corpus_texts[idx],
            "question": meta["question"],
            "answer": meta["answer"],
            "label": meta["label"],
            "index": idx
        })
    
    # Sort by adjusted score
    results.sort(key=lambda x: x["adjusted_score"], reverse=True)
    
    # Cache results
    if use_cache:
        cache_manager.set(query, results, top_k=top_k, min_score=min_score)
    
    return results, cache_hit

# ============================================================
# RATIO CONTEXT BUILDER
# ============================================================

def build_ratio_context(stock_ratios: Optional[Dict[str, float]]) -> str:
    """
    Enhanced ratio context with Lynch's interpretation
    """
    if not stock_ratios or not RATIO_DESCRIPTIONS:
        return ""
    
    chunks = []
    
    # Add interpretive header
    chunks.append("Financial Metrics (Lynch's Perspective):")
    
    for key, val in stock_ratios.items():
        desc = RATIO_DESCRIPTIONS.get(key)
        if desc:
            # Add Lynch-style interpretation
            interpretation = ""
            if key == "peg_ratio":
                if val < 1.0:
                    interpretation = " [Attractive]"
                elif val > 2.0:
                    interpretation = " [Overvalued]"
            elif key == "pe_ratio":
                if val < 15:
                    interpretation = " [Value territory]"
                elif val > 40:
                    interpretation = " [Growth premium]"
            
            chunks.append(f"• {key}: {val}{interpretation} — {desc}")
    
    return "\n".join(chunks)

# ============================================================
# ENHANCED TEXT CLEANING (10/10 VERSION)
# ============================================================

def _clean_generated_answer(text: str, min_length: int = None, max_length: int = None) -> str:
    """
    Production-ready cleaning layer for RAG answers.
    Removes prompt leakage, meta-commentary, removes partial sentences,
    deduplicates content, and forces clean sentence endings.
    
    Note: Does not detect factual hallucinations, only removes model artifacts.
    """
    if not text:
        return text
    
    min_length = min_length or config.answer_min_length
    max_length = max_length or config.answer_max_length
    
    original_length = len(text)
    
    # -----------------------------------------------------
    # 1) Remove any leaked prompt / instruction artifacts
    # -----------------------------------------------------
    prompt_artifacts = [
        "Lynch's Knowledge (use this to answer):",
        "Complete Answer (remember: full sentences, specific Lynch principles):",
        "User Question:",
        "CRITICAL INSTRUCTIONS:",
        "Previous conversation:",
        "Your answer MUST",
        "VERY IMPORTANT:",
    ]
    
    # Remove artifacts intelligently - just the phrases, not everything after
    for artifact in prompt_artifacts:
        # First try to remove if it's at the start of a line
        text = re.sub(r'^' + re.escape(artifact) + r'\s*', '', text, flags=re.MULTILINE)
        # Then remove any remaining instances
        text = text.replace(artifact, "")
    
    # Remove speaker tags like "User:" / "LynchBot:" if they leak into the answer
    text = re.sub(r'^(User|LynchBot)\s*:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(User|LynchBot)\s*:\s*', '', text)
    
    # Clean up excessive whitespace from artifact removal
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # -----------------------------------------------------
    # 2) Split into sentences intelligently
    # -----------------------------------------------------
    if USE_NLTK:
        # Better handling of abbreviations, decimals, etc.
        raw_sentences = nltk.sent_tokenize(text)
    else:
        # Fallback regex - note: may break on "Mr. Lynch" or "15.5. That's good"
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        logger.debug("Using regex sentence splitter - consider installing nltk for better accuracy")
    
    # -----------------------------------------------------
    # 3) Clean duplicates and filter meta-commentary
    # -----------------------------------------------------
    cleaned_sentences = []
    seen_normalized = set()
    
    # More specific meta patterns that won't catch legitimate finance content
    META_PATTERNS = [
        "as an ai language model",
        "as a language model",
        "i am an ai",
        "i do not have access to real-time",
        "i do not have access to the internet",
        "i was trained on",
        "i cannot provide financial advice",
        "my training data",
        "according to my training",
    ]
    
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        
        # Normalize for duplicate detection
        norm = re.sub(r'[^a-zA-Z0-9 ]+', '', s.lower())
        
        # Skip exact duplicates
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)
        
        # Skip meta/model-leak sentences (using full phrases, not single words)
        s_lower = s.lower()
        if any(pattern in s_lower for pattern in META_PATTERNS):
            logger.debug(f"Filtered meta sentence: {s[:50]}...")
            continue
        
        # Additional check: skip very short fragments
        if len(s) < 10 and not re.search(r'[.!?]$', s):
            continue
        
        cleaned_sentences.append(s)
    
    if not cleaned_sentences:
        logger.warning("All sentences filtered out - returning empty")
        return ""
    
    # -----------------------------------------------------
    # 4) Handle incomplete last sentence intelligently
    # -----------------------------------------------------
    if cleaned_sentences:
        last = cleaned_sentences[-1]
        
        # Check if last sentence has proper ending punctuation
        if not re.search(r'[.!?]$', last):
            # If it's substantial (not a fragment), just add punctuation
            if len(last) > 20:
                # Complete thought missing punctuation - fix it
                cleaned_sentences[-1] = last.rstrip('.!?,;:') + '.'
            else:
                # Short fragment - drop it
                cleaned_sentences = cleaned_sentences[:-1]
                
                # But ensure we still have content
                if not cleaned_sentences and len(last) > 10:
                    # If that was our only sentence and it's not tiny, keep it
                    cleaned_sentences = [last.rstrip('.!?,;:') + '.']
    
    # -----------------------------------------------------
    # 5) Join everything and apply length constraints
    # -----------------------------------------------------
    final = " ".join(cleaned_sentences).strip()
    
    # Ensure final output ends with proper punctuation
    if final and final[-1] not in ".!?":
        final += "."
    
    # Apply length constraints
    if len(final) < min_length:
        logger.warning(f"Response too short ({len(final)} chars) - below minimum {min_length}")
        return ""  # Or return a fallback message
    
    if len(final) > max_length:
        # Truncate at word boundary to avoid mid-word cuts
        truncated = final[:max_length].rsplit(' ', 1)[0]
        # Ensure proper ending
        if truncated[-1] not in '.!?':
            truncated += '...'
        final = truncated
        logger.debug(f"Response truncated from {len(final)} to {max_length} chars")
    
    # -----------------------------------------------------
    # 6) Final safety check for aggressive filtering
    # -----------------------------------------------------
    if original_length > 100 and len(final) < original_length * 0.3:
        logger.warning(
            f"Aggressive filtering detected: {original_length} → {len(final)} chars "
            f"({100 * len(final) / original_length:.1f}% retained)"
        )
    
    return final

# ============================================================
# POST-GENERATION VALIDATION
# ============================================================

def post_generation_validation(answer: str, query: str) -> str:
    """
    Simple validation to fix known issues in generated answers:
    - strip URLs / fake sources
    - patch Sharpe ratio misdefinition
    - patch circular PEG definitions
    - strip made-up ISO dates
    """
    if not answer:
        return answer
    
    q_lower = query.lower()
    
    # Remove any URLs (hallucinated sources)
    answer = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '',
        answer
    )
    answer = re.sub(r'www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}', '', answer)
    answer = re.sub(r'\(Source:.*?\)', '', answer)
    answer = re.sub(r'__.*?__', '', answer)  # Remove __text__ formatting
    
    # Fix known bad Sharpe ratio answer (confusing with current ratio)
    if "sharpe ratio" in q_lower and "short-term liabilities" in answer.lower():
        return (
            "Sharpe ratio measures risk-adjusted return: excess return divided by volatility. "
            "Lynch himself didn’t rely heavily on Sharpe; he preferred simple, company-specific "
            "metrics like earnings growth, the PEG ratio, and balance-sheet strength."
        )
    
    # Fix circular PEG definitions
    if "peg" in q_lower and "PEG takes the traditional PEG" in answer:
        return (
            "The PEG ratio is the price-to-earnings (P/E) ratio divided by the company’s earnings "
            "growth rate. Lynch liked PEG because it adjusts valuation for growth: a PEG around 1 "
            "suggests a stock is fairly priced for its growth, while below 1 can be attractive."
        )
    
    # Remove obvious hallucinations about exact ISO dates (YYYY-MM-DD)
    answer = re.sub(r'\d{4}-\d{2}-\d{2}', '', answer)
    
    # Clean up multiple spaces left by removals
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Ensure answer starts with capital letter
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    
    # Ensure answer ends with proper punctuation
    if answer and answer[-1] not in '.!?':
        answer += '.'
    
    return answer

# ============================================================
# PROMPT ENGINEERING (SIMPLIFIED CONVERSATIONAL PROMPT)
# ============================================================

def _build_prompt(
    query: str,
    retrieved_qas: List[Dict],
    ratio_context: str = "",
    chat_context: str = "",
    user_profile: Optional[Dict] = None
) -> str:
    """
    Simpler, conversational prompt for LynchBot.
    Keeps answers grounded in context but sounds like a natural conversation.
    """
    query = (query or "").strip()

    # Fallback when nothing relevant is found
    if not retrieved_qas:
        return f"""
You are LynchBot, a friendly assistant who explains Peter Lynch's investing philosophy in simple, conversational language.

There isn't a direct match for this question in the Peter Lynch notes we have.
Instead, gently guide the user toward topics Lynch is known for, like:
- how he looks for growth and earnings,
- "invest in what you know",
- tenbaggers,
- how he thinks about fundamentals and valuation.

User question: {query}

Answer in a natural, conversational way in 4–6 full sentences. 
Be honest if we don't have specific details, and don't guess.
""".strip()

    # Build context from the top retrieved answers (avoid near-duplicates).
    # Only use top 3 for prompt context to avoid truncation.
    answer_blocks = []
    seen_hashes = set()
    MAX_CONTEXT_CHARS_PER_ANSWER = 450

    for qa in retrieved_qas[:3]:
        ans = qa["answer"].strip()
        if not ans:
            continue

        # Optional: truncate very long answers in the prompt context
        if len(ans) > MAX_CONTEXT_CHARS_PER_ANSWER:
            temp = ans[:MAX_CONTEXT_CHARS_PER_ANSWER]
            ans = temp.rsplit(" ", 1)[0] + "..."

        h = hashlib.md5(ans.lower()[:120].encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        answer_blocks.append(ans)

    context_str = "\n\n".join(answer_blocks)

    # Optional blocks
    ratio_block = f"\n{ratio_context}\n\n" if ratio_context.strip() else ""
    history_block = f"{chat_context}" if chat_context.strip() else ""

    profile_block = ""
    if user_profile:
        style = user_profile.get("investment_style") or ""
        exp = user_profile.get("experience_level") or ""
        if style or exp:
            profile_block = f"User profile: {style} investor, {exp} experience.\n\n"

    # Final prompt: short, clear, conversational
    prompt = f"""
You are LynchBot, a friendly assistant who explains Peter Lynch's investing ideas in a clear, conversational way.

Below is background information drawn from Peter Lynch's own explanations and notes. 
Use ONLY this information when you answer. If something is not covered in the context, say so instead of guessing.

{profile_block}{history_block}{ratio_block}Context:
{context_str}

User question: {query}

Now reply as if you are chatting with an interested investor. 
Give ONE well-structured answer in 4–7 complete sentences (no bullet points), 
be specific but easy to understand, and make sure you finish your explanation cleanly 
without cutting off mid-sentence.

Answer:
""".strip()

    return prompt

# ============================================================
# GENERATION
# ============================================================

def _validate_answer_completeness(text: str) -> bool:
    """
    Simple check if the generated answer is complete
    """
    if not text or len(text) < 50:
        return False
    
    # Check for proper ending punctuation
    last_char = text.strip()[-1] if text.strip() else ''
    if last_char not in '.!?"\'':
        return False
    
    return True

def _generate_text(
    prompt: str,
    min_length: Optional[int] = None,
    temperature: Optional[float] = None,
    max_retries: int = 1
) -> str:
    """
    Enhanced generation with completeness validation
    """
    min_length = min_length or config.min_length
    temperature = temperature or config.temperature
    
    best_answer = ""
    
    for attempt in range(max_retries):
        try:
            # Encode prompt
            input_ids = gen_tokenizer.encode(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(config.device)
            
            # Adjust parameters for each retry
            current_min = min_length + (attempt * 20)
            current_max = config.max_length + (attempt * 50)
            
            # Generate with optimized parameters
            with torch.no_grad():
                output_ids = gen_model.generate(
                    input_ids,
                    max_length=current_max,
                    min_length=current_min,
                    num_beams=config.num_beams + attempt,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    repetition_penalty=config.repetition_penalty,
                    length_penalty=config.length_penalty,
                    temperature=temperature,
                    do_sample=temperature > 0.1,
                    top_p=0.95,
                    early_stopping=False if attempt > 0 else True,
                )
            
            # Decode
            raw = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            
            # Clean the output with enhanced cleaning
            cleaned = _clean_generated_answer(raw)
            
            # Validate completeness
            if _validate_answer_completeness(cleaned):
                logger.debug(f"Generated complete answer on attempt {attempt + 1}")
                return cleaned
            
            best_answer = cleaned
            logger.warning(f"Incomplete answer on attempt {attempt + 1}, retrying...")
            
        except Exception as e:
            logger.error(f"Generation error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
    
    # Return best attempt if all failed
    return best_answer if best_answer else "I need more context from the Lynch dataset to provide a complete answer."

# ============================================================
# PUBLIC API
# ============================================================

def answer_question(
    query: str,
    stock_ratios: Optional[Dict[str, float]] = None,
    chat_history: Optional[List[Dict]] = None,
    session_id: Optional[str] = None,
    user_profile: Optional[Dict] = None,
    return_metadata: bool = False
):
    """
    Main entry point for the RAG chatbot with full feature set
    
    Returns either str or Tuple[str, Dict] based on return_metadata flag
    """
    try:
        query_raw = (query or "").strip()
        query_raw = normalize_query_spelling(query_raw)
        if not query_raw:
            return "Please enter a valid question about Peter Lynch's investment philosophy."
        
        # Log query if enabled
        if config.log_queries:
            logger.info(f"Query: {query_raw[:100]}...")
        
        # Retrieve relevant Q&As (internal function with cache info)
        retrieved, cache_hit = _retrieve_relevant_qa_with_cache_info(query_raw)
        
        if not retrieved:
            logger.info("No relevant matches found")
            fallback = (
                "I couldn't find specific information about that in the Peter Lynch dataset. "
                "Please ask about his investment philosophy, stock selection strategies, "
                "the PEG ratio, growth investing, or fundamental analysis."
            )
            return (fallback, {"retrieved_count": 0, "cache_hit": False}) if return_metadata else fallback
        
        # Log retrieval results
        logger.info(f"Retrieved {len(retrieved)} relevant documents (cache hit: {cache_hit})")
        for r in retrieved[:3]:
            logger.debug(f"  Score: {r['adjusted_score']:.3f} (boost: {r['boost']:.3f}) | Label: {r['label']}")
        
        # Build context components
        ratio_context = build_ratio_context(stock_ratios)
        
        # Handle conversation history
        chat_context = ""
        if config.use_chat_history_in_prompt:
            if session_id and session_id != "default":
                chat_context = conversation_manager.format_history(session_id)
            elif chat_history:
                # Format provided history
                lines = []
                for turn in chat_history[-8:]:  # Last 8 turns
                    role = "LynchBot" if turn.get("role") == "assistant" else "User"
                    content = turn.get("content", "").strip()
                    if content:
                        lines.append(f"{role}: {content}")
                if lines:
                    chat_context = "Previous conversation:\n" + "\n".join(lines) + "\n\n"
        
        # Build prompt (simplified conversational version)
        prompt = _build_prompt(
            query=query_raw,
            retrieved_qas=retrieved,
            ratio_context=ratio_context,
            chat_context=chat_context,
            user_profile=user_profile
        )
        
        # Generate answer
        final_answer = _generate_text(prompt)
        
        # Post-generation validation to strip links & patch known bad patterns
        final_answer = post_generation_validation(final_answer, query_raw)
        
        # Update conversation history if session provided
        if session_id:
            conversation_manager.add_turn(session_id, "user", query_raw)
            conversation_manager.add_turn(session_id, "assistant", final_answer)
        
        # Prepare metadata if requested
        if return_metadata:
            metadata = {
                "retrieved_count": len(retrieved),
                "top_scores": [r["adjusted_score"] for r in retrieved[:3]],
                "boosts": [r["boost"] for r in retrieved[:3]],
                "labels": list(set(r["label"] for r in retrieved if r["label"])),
                "cache_hit": cache_hit,
                "session_id": session_id
            }
            return final_answer, metadata
        
        return final_answer
        
    except Exception as e:
        logger.error(f"Error in answer_question: {e}", exc_info=True)
        error_msg = "I encountered an error processing your question. Please try again."
        if return_metadata:
            return error_msg, {"error": str(e)}
        return error_msg

def answer_conversational(
    query: str,
    session_id: str = "default",
    stock_ratios: Optional[Dict[str, float]] = None,
    user_profile: Optional[Dict] = None
) -> str:
    """
    Conversational interface with automatic session management
    """
    return answer_question(
        query=query,
        stock_ratios=stock_ratios,
        session_id=session_id,
        user_profile=user_profile,
        return_metadata=False
    )

# Backward compatibility
get_answer = answer_question
run_rag = answer_question
MODEL_READY = True

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clear_cache():
    """Clear the query cache"""
    cache_manager.clear()

def clear_session(session_id: str = "default"):
    """Clear conversation history for a session"""
    conversation_manager.clear_session(session_id)

def get_statistics() -> Dict:
    """Get pipeline statistics"""
    return {
        "total_qa_pairs": len(df),
        "unique_labels": df["Label"].nunique() if "Label" in df.columns else 0,
        "cache_entries": len(cache_manager.cache),
        "active_sessions": len(conversation_manager.sessions),
        "model_device": config.device,
        "semantic_boost_enabled": config.enable_semantic_boost,
        "nltk_available": USE_NLTK,
        "use_chat_history_in_prompt": config.use_chat_history_in_prompt,
    }

# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    print("\n🎯 Lynch RAG Pipeline v2.2 + Validation - Interactive Mode")
    print("=" * 60)
    print(f"📊 Statistics: {get_statistics()}")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'clear' - Clear conversation history")
    print("  'clear cache' - Clear query cache")
    print("  'stats' - Show statistics")
    print("  'ratios' - Enter stock ratios mode")
    print("-" * 60)
    
    session_id = "cli_session"
    stock_ratios = None
    
    while True:
        try:
            user_input = input("\n💬 Question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                clear_session(session_id)
                print("✅ Conversation cleared")
                continue
            
            elif user_input.lower() == 'clear cache':
                clear_cache()
                print("✅ Cache cleared")
                continue
            
            elif user_input.lower() == 'stats':
                print(f"📊 {get_statistics()}")
                continue
            
            elif user_input.lower() == 'ratios':
                print("Enter stock ratios (e.g., pe_ratio=15.5) or 'done':")
                ratios = {}
                while True:
                    ratio_input = input("  Ratio: ").strip()
                    if ratio_input.lower() == 'done':
                        break
                    if '=' in ratio_input:
                        key, val = ratio_input.split('=')
                        try:
                            ratios[key.strip()] = float(val.strip())
                        except ValueError:
                            print("  Invalid format. Use: key=value")
                stock_ratios = ratios if ratios else None
                print(f"✅ Ratios set: {stock_ratios}")
                continue
            
            # Process the question
            print("\n🤖 LynchBot:")
            answer = answer_conversational(
                user_input,
                session_id=session_id,
                stock_ratios=stock_ratios
            )
            
            print(f"\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception("Error in CLI")
