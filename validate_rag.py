# validate_rag.py
"""
Smoke test for the RAG pipeline. Run after setup to confirm the dataset loads,
the models come up, and generation actually returns a grounded answer:

    python validate_rag.py
"""

import sys

from rag_pipeline import USE_NLTK, answer_conversational, get_statistics

QUESTIONS = [
    "Why is AAPL a long-term growth stock?",
    "What is the PEG ratio and why did Lynch like it?",
]

# answer_question traps exceptions and returns these instead of raising, so a
# smoke test has to match on them to notice a failure.
FAILURE_MARKERS = (
    "I encountered an error processing your question",
    "I need more context from the Lynch dataset",
)


def main() -> int:
    print(f"Statistics: {get_statistics()}")
    if not USE_NLTK:
        print(
            "NOTE: NLTK sentence tokenizer unavailable, falling back to regex. "
            "Install it with: python -m nltk.downloader punkt_tab"
        )

    failures = 0
    for question in QUESTIONS:
        answer = answer_conversational(question, session_id="validate_rag")
        print(f"\nQ: {question}\nA: {answer}")

        if not answer or any(marker in answer for marker in FAILURE_MARKERS):
            print("   -> FAILED")
            failures += 1

    print(f"\n{len(QUESTIONS) - failures}/{len(QUESTIONS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
