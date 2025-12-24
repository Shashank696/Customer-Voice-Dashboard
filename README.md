# 🗣️ Customer Voice Insight Dashboard

An AI-powered dashboard that automatically analyzes customer feedback to identify hidden pain points. Built with **Transformers (RoBERTa)** and **Unsupervised Learning (HDBSCAN)**.

## 🚀 Features
* **Sentiment Analysis:** Filters negative reviews using a pre-trained RoBERTa model.
* **Topic Modeling:** Groups complaints into clusters using Embeddings + UMAP + HDBSCAN.
* **Auto-Labeling:** Uses Generative AI (Flan-T5) to name the topic clusters automatically.
* **Interactive UI:** Built with Streamlit for real-time data exploration.

## 🛠️ Tech Stack
* **Python** (Pandas, NumPy)
* **Hugging Face Transformers** (RoBERTa, Flan-T5)
* **Sentence-Transformers** (GTE-Small)
* **Clustering:** UMAP, HDBSCAN
* **Visualization:** Plotly, Streamlit

## 📖 Book Reference
This project implements concepts from *Hands-On Large Language Models* (Alammar & Grootendorst):
* **Chapter 4:** Text Classification (Sentiment)
* **Chapter 5:** Clustering & Topic Modeling

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Shashank696/Customer-Voice-Dashboard.git](https://github.com/Shashank696/Customer-Voice-Dashboard.git)