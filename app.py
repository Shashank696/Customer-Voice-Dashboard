import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Voice Dashboard", layout="wide")

st.title("🗣️ Customer Voice Insight Dashboard")
st.markdown("""
This app replicates the logic from **Chapter 4 & 5** of the LLM Book:
1. **Sentiment Analysis** (RoBERTa) to find negative reviews.
2. **Topic Modeling** (Embeddings + UMAP + HDBSCAN) to group complaints.
3. **Generative AI** (Flan-T5) to name the topics automatically.
""")

# --- 1. LOAD MODELS (CACHED) ---
# We use @st.cache_resource so models load only ONCE, not every time you click a button.

@st.cache_resource
def load_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True,
        device=device,
        truncation=True,
        max_length=512
    )

@st.cache_resource
def load_embedding_model():
    # You used 'thenlper/gte-small' in your notebook
    return SentenceTransformer("thenlper/gte-small")

@st.cache_resource
def load_labeling_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

# --- 2. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Reviews CSV", type="csv")

if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        st.success(f"Loaded {len(df)} rows.")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    # Allow user to pick the text column
    text_col = st.selectbox("Select the column containing review text:", df.columns, index=0)
    
    if st.button("Start Analysis"):
        
        # --- PHASE 1: SENTIMENT ---
        st.header("1. Sentiment Analysis (RoBERTa)")
        
        pipe = load_sentiment_pipeline()
        
        # Run inference with a progress bar
        texts = df[text_col].astype(str).tolist()
        batch_size = 32 # Smaller batch size is safer for generic web apps
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Your batch processing logic
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            output = pipe(batch, truncation=True, max_length=512)
            results.extend(output)
            
            # Update progress
            progress = min((i + batch_size) / len(texts), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing sentiment... {int(progress*100)}%")
            
        status_text.empty()
        
        # Extract Labels
        labels = [max(res, key=lambda x: x["score"])["label"] for res in results]
        df["sentiment"] = labels
        
        # Show Sentiment Chart
        fig_sentiment = px.pie(df, names='sentiment', title='Sentiment Distribution', hole=0.4)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # --- PHASE 2: TOPIC MODELING ---
        st.header("2. Negative Topic Discovery")
        
        # Filter Negative
        df_neg = df[df["sentiment"] == "negative"].copy().reset_index(drop=True)
        
        if len(df_neg) < 5:
            st.warning("Not enough negative reviews to cluster.")
        else:
            st.info(f"Analyzing {len(df_neg)} negative reviews...")
            
            # A. Embeddings
            embedder = load_embedding_model()
            embeddings = embedder.encode(df_neg[text_col], batch_size=64, show_progress_bar=True)
            
            # B. UMAP
            umap_model = UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42)
            reduced_embeddings = umap_model.fit_transform(embeddings)
            
            # C. HDBSCAN (Your specific settings)
            hdbscan_model = HDBSCAN(
                min_cluster_size=15,
                min_samples=5,
                metric="euclidean",
                cluster_selection_method="eom"
            ).fit(reduced_embeddings)
            
            df_neg["Topic"] = hdbscan_model.labels_
            
            # Add Coordinates for Plotting
            df_neg["x"] = reduced_embeddings[:, 0]
            df_neg["y"] = reduced_embeddings[:, 1]
            
            # --- PHASE 3: AUTO LABELING ---
            st.header("3. AI Topic Labeling (Flan-T5)")
            
            labeler = load_labeling_model()
            topic_names = {}
            unique_topics = sorted(list(set(df_neg["Topic"])))
            
            # Progress bar for labeling
            label_progress = st.progress(0)
            
            for i, topic_id in enumerate(unique_topics):
                if topic_id == -1:
                    topic_names[-1] = "Noise / Outliers"
                else:
                    # Get top 5 reviews
                    reviews_sample = df_neg[df_neg["Topic"] == topic_id][text_col].head(5).tolist()
                    reviews_text = "\n".join([f"- {r}" for r in reviews_sample])
                    
                    # Prompt (Your exact prompt)
                    prompt = f"I have the following customer reviews:\n{reviews_text}\n\nBased on these reviews, give me a short, 3-word topic label:"
                    
                    generated = labeler(prompt, max_length=20)[0]['generated_text']
                    topic_names[topic_id] = f"{topic_id}: {generated}"
                
                label_progress.progress((i + 1) / len(unique_topics))
            
            # Map names
            df_neg["Topic Name"] = df_neg["Topic"].map(topic_names)
            
            # --- PHASE 4: FINAL VISUALIZATION ---
            st.subheader("Topic Cluster Map")
            
            fig_clusters = px.scatter(
                df_neg, 
                x="x", y="y", 
                color="Topic Name", 
                hover_data=[text_col],
                title="Negative Review Clusters (Interactive)"
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            st.subheader("Data Table")
            st.dataframe(df_neg[[text_col, "sentiment", "Topic Name"]])