# validate_rag.py
from rag_pipeline import rag_answer

q = "Why is AAPL a long-term growth stock?"
print(rag_answer(q))
