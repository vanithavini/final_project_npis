## 📰 News Popularity Intelligence System using Transformer-Based Deep Learning ##

## 📌 Project Overview
- Digital news platforms must prioritize articles before real-world engagement signals (clicks, shares, impressions, CTR) become    available.
- This project builds an Explainable Transformer-Based Intelligence System that predicts the popularity potential of news articles   using:
    - Transformer-based semantic representation (DistilBERT)
    - Linguistic proxy intelligence signals
    - Weighted editorial scoring framework
    - Explainable AI (XAI) reasoning layer
    - Priority-based ranking engine
    - Interactive Streamlit dashboard
- Unlike traditional supervised models, this system treats popularity as a latent variable inferred through proxy indicators, making it a weakly supervised intelligence framework.

## 🧩 Problem Statement
- Digital news platforms must decide which articles to highlight, promote, or deprioritize at the time of publishing. However, real-world popularity indicators such as clicks, shares, and impressions are not immediately available.
- The objective of this project is to design and implement a Transformer-based News Popularity Intelligence System that:
    * Learns deep semantic representations of news articles using transfer learning
    * Infers relative popularity potential directly from text
    * Ranks and scores articles based on predicted attention likelihood
    * Provides explainable insights to support editorial decision-making

## 🏗 System Architecture
The system follows an enterprise-layered AI architecture:

Data Layer
   ↓
NLP Representation Layer (DistilBERT)
   ↓
Signal Intelligence Layer
   ↓
Weighted Scoring Engine
   ↓
Decision & Explainability Layer
   ↓
Streamlit Intelligence Dashboard

# 🔵 Data Layer
Raw news dataset
Title + Description extraction
Text preprocessing

# 🟢 NLP Representation Layer
Text cleaning
DistilBERT embeddings
Semantic vector representation

# 🟠 Intelligence & Signal Layer
Proxy signals used to estimate popularity:
Emotional Intensity
Urgency
Lexical Diversity
Readability
Subjectivity
Length Signal

# 🟣 Decision Layer
Weighted aggregation
Popularity percentage score
Priority classification (High / Medium / Low)
Article ranking
Explainability engine

## 🧠 Scoring Logic
Final popularity score is computed as:

Final Score =
    (Emotion * 0.25) +
    (Urgency * 0.20) +
    (Lexical Diversity * 0.15) +
    (Readability * 0.15) +
    (Length * 0.10) +
    (Subjectivity * 0.15)

The weights reflect editorial influence assumptions and can be tuned.

## 🔍 Explainability Framework
The Explainability Engine provides:
Contribution breakdown per signal
Top contributing factors
Transparent scoring logic
Contribution formula:
    Contribution = Signal Score × Weight × 100
This ensures interpretability and trust in decision-making.

## 📊 Features
✔ Transformer-based semantic analysis
✔ Weakly supervised popularity inference
✔ Modular AI architecture
✔ Weighted editorial intelligence
✔ Explainable AI (XAI)
✔ Article ranking engine
✔ Interactive Streamlit dashboard
✔ Architecture visualization using Graphviz

## 🗂 Project Structure
news-popularity-intelligence/
│
├── data/
│   ├── raw/
│   │    ├── news.csv
│   └── processed/
│        ├── embeddings_chunks
│        ├── news_cleaned.csv
│        ├── news_embeddings.npy
│        ├── news_popularity_scored.csv
│
├── notebooks/
│   ├── 01_raw_data_sanity_check.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_bert_representation_learning.ipynb
│   ├── 04_popularity_scoring.ipynb
│   ├── 05_article_ranking.ipynb
│   └── 06_explainability_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── embedding_generator.py
│   ├── explainability_engine.py
│   ├── popularity_scorer.py
│   ├── popularity_signals.py
│   ├── ranking_engine.py
│
├── app.py
├── requirements.txt
└── README.md

## 🔍 Module-wise Description
# Module 1
  1.1 --> raw_data_sanity_check.ipynb    # Convert raw, messy news text into a clean, model-ready textual input while: Preserving
  1.2 --> data_preprocessing.py         semantics, avoiding over-processing, maintaining traceability for explainability. EDA is used
  1.3 --> eda.ipynb                     for assumption validation rather than correlation analysis because of no labels exists.

# Module 2
  2.1 --> embedding_generator.py                # Convert each news article into a dense semantic vector. Capture - emotion, urgency, 
  2.2 --> bert_representation_learning.ipynb    semantics, narrative style. Save the embeddings for downstream scoring and ranking.

# Module 3
  3.1 --> popularity_signals.py          # We designed a weakly-supervised popularity scoring engine that infers attention likelihood
  3.2 --> popularity_scorer.py           using editorial signals.The system normalize scores into 0-100 scale & classifies articles 
  3.3 --> popularity_scoring.ipynb       into priority tiers, enabling explainable & label-free ranking. Popularity is latent. we 
                                         infer it using attention-related signals.

# Module 4
  4.1 --> ranking_engine.py               # Rank articles using BERT embeddings(semantic strength) & Popularity Score(attention
  4.2 --> article_ranking.ipynb            likelihood). Generate clear,human-readable explanations.Explainability is achieved by 
  4.3 --> explainability_engine.py         decomposing the popularity score into weighted linguistic signal contributions, enabling
  4.4 --> explainability_analysis.ipynb    transparent editorial reasoning.

# Module 5
  Streamlit app --> app.py                # We implemented comparitive explainability by visualizing signal-level contribution 
                                           differences between emotionally intense and neutral articles.

## 🚀 Installation & Setup
1️⃣ Clone Repository
    git clone 

2️⃣ Create Virtual Environment
    python -m venv .venv
    .venv\Scripts\activate      

3️⃣ Install Dependencies
    pip install -r requirements.txt

▶️ Run the Application
    streamlit run app.py

The dashboard will open in your browser.

## 📦 Dependencies
Key dependencies used:
- Data Handling
        pandas
        numpy
- NLP & Linguistic Analysis
        nltk
        textblob
        textstat
- Transformer Models
        torch
        transformers
        sentencepiece
- Machine Learning Utilities
        scikit-learn
- Visualization
        matplotlib
        seaborn
        graphviz
- Deployment
        streamlit

## 📈 Example Use Case
# Article 1
"Breaking: Massive earthquake devastates coastal city"
    High emotion
    High urgency
    High popularity score

# Article 2
"Quarterly economic statistics report released"
    Low emotion
    Low urgency
    Lower popularity score

The system ranks  Article 1 higher due to stronger proxy signals.

## 🎯 Why This Project Is Unique
- Does not rely on labeled popularity data
- Models popularity as a latent variable
- Combines Transformer NLP + heuristic intelligence
- Fully explainable scoring framework
- Enterprise-style layered architecture
- Deployable AI dashboard

## 🏢 Industry Applications
- Digital news platforms
- Editorial prioritization systems
- Content recommendation engines
- Media analytics platforms
- Publishing workflow automation

## 🔮 Future Improvements
- Fine-tuned Transformer model for engagement prediction
- Reinforcement learning for weight optimization
- Real-time API deployment
- A/B testing integration
- User personalization layer
- Automated weight learning via weak supervision

