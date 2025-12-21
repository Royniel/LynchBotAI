Peter Lynch chatbot

### 1\. Scope and current status

Part I -- Chat-bot (RAG + dataset engineering)

-   Implemented a Retrieval-Augmented Generation (RAG) chatbot that answers questions in the style of Peter Lynch.

-   The chatbot is backed by a curated and engineered Q&A dataset (~1,500+ rows), organized into six labels:

-   Personal Life

-   Strategy Development

-   Timing

-   Risk Management

-   Adaptability

-   Psychology

-   The dataset involved several non-trivial preprocessing and design steps:

-   Integrated the original Lynch dataset with multiple additional batches of custom Q&A pairs to improve coverage and realism.

-   Removed all duplicate Q--A--Label combinations across the full corpus to avoid training and retrieval bias.

-   Standardized labels into the final six-category taxonomy, ensuring consistent semantics across all entries.

-   Identified label imbalance (e.g., Risk Management and Adaptability being underrepresented) and addressed it by generating additional high-quality Q&As focused on:

-   Black swan events and tail risk

-   Different interest-rate regimes

-   Position sizing under uncertainty

-   Adapting to macro shifts, tech disruption, and structural change

-   The dataset was deliberately written with a RAG use case in mind:

-   Mix of clean, conceptual questions and more conversational, messy, "real user" questions.

-   Coverage of not just theory but scenarios, emotional reactions, and practical decision-making.

-   All answers framed in a way that reflects Peter Lynch--style reasoning and language, so the model can stay in-character while still being practical.

-   RAG pipeline:

-   Embedded the Q&A corpus using a sentence-transformer model to enable semantic retrieval.

-   At query time, the system retrieves the most relevant Q&As (using cosine similarity on embeddings) and passes them, along with the user's question, into a generative model (T5-style).

-   The model then composes a final answer that blends:

-   The retrieved Lynch-style knowledge, and

-   The specific details of the user's query.

This makes Part I more than a simple FAQ: it is a structured knowledge base plus a retrieval and generation pipeline tuned for an investing mentor persona.

* * * * *

Part II -- Stock integration (data, modeling, and UI)

-   Implemented a full stock-analysis layer that is tightly integrated with the chatbot UI:

-   Defined a default Dow Jones portfolio and built logic for users to add and remove tickers dynamically.

-   Developed a dedicated data ingestion pipeline using yfinance, including:

-   Handling of missing fields and NaNs.

-   Defensive logic around tickers that return incomplete or inconsistent data.

-   Normalization of raw Yahoo Finance fields into a consistent internal schema suitable for ratio computation.

-   Feature engineering and modeling:

-   Engineered a set of value and quality ratios (e.g., valuation, profitability, leverage, efficiency) and assembled them into a structured fin_data_df table.

-   Preprocessed the features (selection, scaling, NaN handling) to make them suitable for K-Means.

-   Applied K-Means clustering to segment stocks into interpretable value/quality clusters.

-   Built logic to derive long vs short recommendation lists from the cluster structure, so the system outputs actionable groupings rather than just raw labels.

-   UI and integration:

-   Exposed a dashboard section that presents the financial ratios DataFrame for all selected stocks.

-   Surfaces cluster-aware long/short recommendations directly in the UI.

-   Wired the chatbot so that responses can be grounded not only in the Lynch Q&A corpus but also in:

-   The current portfolio composition,

-   The computed ratios, and

-   The cluster-based characterization of each stock.

Overall, the stock integration is a complete mini-pipeline: data ingestion → cleaning → feature engineering → clustering → interpretation → UI integration, rather than just a single API call.

* * * * *

### 2\. High-level architecture

-   RAG engine

-   Uses a sentence-transformer model to embed all Q&A entries in the Lynch corpus.

-   For each user query, retrieves the top relevant Q&As based on semantic similarity.

-   Combines the retrieved context with the user question and feeds this into a generative model (T5-family) to produce the final answer.

-   Stock analytics

-   Uses yfinance to pull fundamental data for the active set of tickers with basic robustness checks.

-   Constructs a ratios DataFrame for value and quality metrics.

-   Runs K-Means clustering on the ratio features to create a segmentation of the portfolio.

-   Uses cluster membership to derive long/short lists and enable the chatbot to comment on both qualitative philosophy and quantitative signals.

* * * * *

### 3\. How to install and run locally

Clone the repository:

git clone <REPO_URL>

cd <PROJECT_FOLDER>

1.

(Optional but recommended) Create and activate a virtual environment:

# Create venv

python -m venv .venv

# Activate on macOS/Linux

source .venv/bin/activate

# Activate on Windows

.venv\Scripts\activate

1.

Install dependencies:

pip install -r requirements.txt

1.

2.  Place the Lynch dataset file:

-   Ensure the final cleaned dataset (for example lynch_rag_10of10_master_dedup_balanced.xlsx) is located under the data/ directory.

If the code expects a specific filename (for example lynch_dataset.xlsx), rename the file accordingly:

mv data/lynch_rag_10of10_master_dedup_balanced.xlsx data/lynch_dataset.xlsx

Run the Streamlit application:

streamlit run app.py

1.  Then open the URL shown in the terminal (typically http://localhost:8501).

* * * * *

### 4\. What to review

-   Interact with the chatbot using:

-   Strategy, timing, and risk questions.

-   Psychology and personal-life-related questions (how lifestyle, behaviour, or mindset affect investing).

-   Adaptability-focused questions (how to adjust to macro shifts, tech changes, and different regimes).

-   Modify the stock selection (add/remove tickers) and verify:

-   The financial ratios table updates as expected.

-   K-Means clusters and long/short recommendations refresh correctly.

-   The chatbot's responses reflect both Lynch-style reasoning and the updated portfolio context.

If helpful, I can also walk through the RAG pipeline, dataset design choices, and the financial/clustering implementation in a short session.
