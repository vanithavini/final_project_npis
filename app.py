import streamlit as st
import pandas as pd

from src.popularity_scorer import PopularityScorer
from src.explainability_engine import ExplainabilityEngine
from src.ranking_engine import ArticleRankingEngine


# Initialize Engines
scorer = PopularityScorer()
explainer = ExplainabilityEngine(scorer.weights)
ranking_engine = ArticleRankingEngine()


st.set_page_config(page_title="News Popularity Intelligence System", layout="wide")

st.title("📰 News Popularity Intelligence System")
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "News Intelligence", "Model Reasoning"]
)

# 1️⃣ HOME PAGE
###############
if page == "Home":
    st.header("Project Overview")

    st.write("""
    Digital news platforms must prioritize articles before real-world popularity signals (clicks, shares) become available.

    This system predicts popularity potential using:
    - Transformer-based semantic understanding
    - Linguistic proxy signals
    - Weighted scoring logic
    - Explainable ranking mechanisms
    """)

    st.subheader("System Architecture")

    st.markdown("""
    1. Text Input (Title + Description)
    2. Linguistic Signal Extraction
    3. Weighted Popularity Scoring
    4. Priority Classification
    5. Ranking & Explainability
    """)

    st.write("""
    Popularity labels such as clicks, shares, and engagement metrics 
    are not available at prediction time. Therefore, we model 
    popularity as a latent variable inferred through linguistic signals.
    """) 

# 2️⃣ NEWS INTELLIGENCE
########################
elif page == "News Intelligence":

    st.header("News Popularity Prediction")

    title = st.text_input("Enter News Title")
    description = st.text_area("Enter News Description")

    if st.button("Analyze Popularity"):

        if title.strip() == "" and description.strip() == "":
            st.warning("Please enter at least a title or description.")
        else:
            text = title + ". " + description

            final_score, signal_scores = scorer.score_article(text)
            label = scorer.priority_label(final_score)

            contributions = explainer.explain(signal_scores)
            top_reasons = explainer.top_reasons(contributions)

            st.subheader("📊 Popularity Results")
            st.metric("Popularity Score", f"{round(final_score, 2)}%")
            st.success(f"Priority Level: {label}")

            st.subheader("🔍 Key Contributing Factors")
            for signal, value in top_reasons:
                st.write(f"**{signal.capitalize()}** → {value}% contribution")

# 3️⃣ MODEL REASONING
#####################
elif page == "Model Reasoning":

    st.header("Model Reasoning & Explainability")

    st.write("""
    Popularity is a latent variable inferred through proxy linguistic indicators.
    """)

    st.subheader("Signals Used")
    st.markdown("""
    - Emotional Intensity
    - Urgency
    - Lexical Diversity
    - Readability
    - Length
    - Subjectivity
    """)

    st.subheader("Scoring Logic")

    st.code("""
Final Score = 
    (Emotion * 0.25) +
    (Urgency * 0.20) +
    (Lexical * 0.15) +
    (Readability * 0.15) +
    (Length * 0.10) +
    (Subjectivity * 0.15)
""")

    st.subheader("Ranking Strategy")

    st.write("""
    Articles are ranked in descending order of normalized popularity score.
    Priority-aware sorting ensures editorial clarity.
    """)

