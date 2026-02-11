import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from src.popularity_scorer import PopularityScorer
from src.explainability_engine import ExplainabilityEngine
from src.ranking_engine import ArticleRankingEngine
from graphviz import Digraph


# Initialize Engines
scorer = PopularityScorer()
explainer = ExplainabilityEngine(scorer.weights)
ranking_engine = ArticleRankingEngine()


st.set_page_config(page_title="News Popularity Intelligence System", layout="wide")

st.title("📰 News Popularity Intelligence System using Transformer-Based Deep Learning")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Home", "News Intelligence", "Model Reasoning"])

# 1️⃣ HOME PAGE
###############
if page == "Home":

    st.header("📌 Project Overview")

    st.write("""
    Digital news platforms must prioritize articles before real-world popularity signals 
    (clicks, shares, impressions) become available.

    This system predicts *popularity potential* using:

    • Transformer-based semantic representation (DistilBERT)  
    • Linguistic proxy intelligence signals  
    • Weighted editorial scoring logic  
    • Explainable ranking framework  
    """)

    st.subheader("🏗 Enterprise System Architecture")

    # Create Diagram
    diagram = Digraph(engine="dot")
    diagram.attr(rankdir="LR", nodesep="0.6", ranksep="0.8")
    diagram.node_attr.update(shape="box", style="filled", fontsize="11", width="1.6", height="0.6")
    diagram.edge_attr.update(fontsize="10")

    # Data Layer
    with diagram.subgraph(name="cluster_data") as c:
        c.attr(label="🔵 Data Layer", style="filled", fillcolor="#E3F2FD", color="#1E88E5", fontcolor="black")
        c.node("A1", "Raw News Dataset", fillcolor="#BBDEFB")
        c.node("A2", "Title + Description", fillcolor="#BBDEFB")
        c.edge("A1", "A2")

    # NLP Layer
    with diagram.subgraph(name="cluster_nlp_v2") as c:
        c.attr(label="🟢 NLP & Representation Layer", style="filled", fillcolor="#E8F5E9", color="#2E7D32", fontcolor="black")
        c.node("B1", "Text Cleaning", fillcolor="#C8E6C9")
        c.node("B2", "DistilBERT Embeddings", fillcolor="#C8E6C9")
        c.node("B3", "Semantic Vector Space", fillcolor="#C8E6C9")
        c.edges([("B1", "B2"), ("B2", "B3")])

    # Intelligence Layer
    with diagram.subgraph(name="cluster_intelligence_v2") as c:
        c.attr(label="🟠 Intelligence & Scoring Engine", style="filled", fillcolor="#FFF3E0", color="#EF6C00", fontcolor="black")
        c.node("C1", "Emotion Signal", fillcolor="#FFE0B2")
        c.node("C2", "Urgency Signal", fillcolor="#FFE0B2")
        c.node("C3", "Lexical Diversity", fillcolor="#FFE0B2")
        c.node("C4", "Readability Score", fillcolor="#FFE0B2")
        c.node("C5", "Subjectivity Score", fillcolor="#FFE0B2")
        c.node("C6", "Length Signal", fillcolor="#FFE0B2")
        c.node("C7", "Weighted Aggregation", fillcolor="#FFCC80")
        c.edges([("C1","C7"), ("C2","C7"), ("C3","C7"), ("C4","C7"), ("C5","C7"), ("C6","C7")])

    # Decision Layer
    with diagram.subgraph(name="cluster_decision_v2") as c:
        c.attr(label="🟣 Decision & Explainability Layer", style="filled", fillcolor="#F3E5F5", color="#6A1B9A", fontcolor="black")
        c.node("D1", "Popularity Score (%)", fillcolor="#E1BEE7")
        c.node("D2", "Priority Classification", fillcolor="#E1BEE7")
        c.node("D3", "Article Ranking", fillcolor="#E1BEE7")
        c.node("D4", "Explainability Engine", fillcolor="#E1BEE7")
        c.edges([("D1","D2"), ("D1","D3"), ("D1","D4")])

    # Application Layer
    diagram.node("E1", "🌐 Streamlit Intelligence Dashboard", fillcolor="#CFD8DC") 

    # Cross-Layer Flow
    diagram.edge("A2", "B1")
    diagram.edge("B3", "C1")
    diagram.edge("C7", "D1")
    diagram.edge("D4", "E1")

    st.graphviz_chart(diagram, use_container_width=True)

    st.markdown("-----------------------------------------------------------------------------")

    st.subheader("📎 Why Popularity Labels Are Unavailable")

    st.write("""
    Real-world engagement signals such as clicks, shares, impressions, and CTR 
    are not available at prediction time.

    Therefore, popularity is treated as a **latent variable** inferred through:
    
    • Emotional intensity  
    • Urgency and novelty  
    • Linguistic clarity  
    • Structural readability  
    • Narrative engagement signals  

    This makes the system a **weakly supervised Transformer-based intelligence framework**, 
    rather than a traditional supervised classification model.
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


    st.subheader("Example Comparison")

    st.write("""
    Example 1:
    "Breaking: Massive earthquake devastates city"
    → High emotion + urgency → High popularity score

    Example 2:
    "Quarterly economic statistics released"
    → Low emotion + low urgency → Lower popularity score
    """)

    st.subheader("📊 Visual Comparison of Articles")

    example_1 = "Breaking: Massive earthquake devastates coastal city, thousands feared affected."
    example_2 = "Quarterly economic statistics report released by government."

    score1, comp1 = scorer.score_article(example_1)
    score2, comp2 = scorer.score_article(example_2)

    comparison_df = pd.DataFrame({
        "Signal": comp1.keys(),
        "Article 1 (Emotional)": comp1.values(),
        "Article 2 (Neutral)": comp2.values()
    })

    st.dataframe(comparison_df)

    # Add Visual bar chart
    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(comparison_df["Signal"], comparison_df["Article 1 (Emotional)"], alpha=0.7, label="Emotional")
    ax.bar(comparison_df["Signal"], comparison_df["Article 2 (Neutral)"], alpha=0.7, label="Neutral")

    plt.xticks(rotation=45)
    plt.ylabel("Signal Strength")
    plt.legend()

    st.pyplot(fig)
