# log_analyzer.py
import streamlit as st
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 AI-Powered Log Analyzer with LLM")

# Simulate fetching logs from backend API or file
def fetch_logs():
    return [
        "[2025-04-01 12:32:45] ERROR auth: Invalid token for user admin",
        "[2025-04-01 12:33:02] WARNING database: Slow query detected",
        "[2025-04-01 12:33:30] INFO server: Started on port 8080",
        "[2025-04-01 12:34:01] ERROR auth: User admin failed login 3 times",
        "[2025-04-01 12:35:12] INFO monitor: CPU usage at 85%",
        "[2025-04-01 12:35:45] ERROR cache: Redis timeout",
        "[2025-04-01 12:36:22] WARNING api: Request took 9s",
        "[2025-04-01 12:37:30] ERROR database: Connection pool exhausted",
        "[2025-04-01 12:38:45] INFO scheduler: Job completed",
    ]

logs = fetch_logs()

st.subheader("📄 Fetched Logs")
st.code("\n".join(logs[:10]), language="bash")

# --- Preprocess logs ---
def preprocess(logs):
    return [re.sub(r'[^a-zA-Z0-9 ]', ' ', log.lower()) for log in logs]

clean_logs = preprocess(logs)

# --- TF-IDF + Clustering ---
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(clean_logs)

n_clusters = st.slider("Number of clusters", 2, 6, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Show clustered logs
st.subheader("🔍 Clustered Logs")
for i in range(n_clusters):
    st.markdown(f"**Cluster {i}**")
    st.code("\n".join([logs[j] for j in range(len(logs)) if labels[j] == i][:5]), language="bash")

# --- Word Cloud ---
st.subheader("☁️ Word Cloud")
wordcloud = WordCloud(width=800, height=300).generate(" ".join(clean_logs))
plt.figure(figsize=(10, 4))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# --- LLM Root Cause Summary ---
st.subheader("🧠 Root Cause Analysis (LLM)")
selected_cluster = st.selectbox("Choose cluster for analysis", list(range(n_clusters)))

cluster_logs = "\n".join([logs[j] for j in range(len(logs)) if labels[j] == selected_cluster])

if st.button("🔎 Analyze with LLM"):
    with st.spinner("Analyzing with LLM..."):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer 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"},
            json={
                "model": "mistralai/mistral-small-3.1-24b-instruct:free",
                "messages": [{
                    "role": "user",
                    "content": f"""
You are an expert log analysis assistant. Given a set of application logs (structured or unstructured), extract actionable insights tailored to the following user personas:

1. Developers
2. DevOps Engineers / Site Reliability Engineers (SREs)
3. QA Testers
4. Security Analysts

For each persona:
- Identify the key log events relevant to them.
- Summarize potential issues or anomalies.
- Provide recommendations or next steps to address those issues.
- Mention specific components (e.g., auth, cache, DB, API) if applicable.

Also, highlight:
- Error patterns
- Latency or performance bottlenecks
- Security-related failures (invalid token, failed login, etc.)
- Repeated issues that may warrant further investigation

Here are the logs:
---
{cluster_text}
---
"""
}],
            }
        )
        result = response.json()
        summary = result['choices'][0]['message']['content']
        st.success("✅ LLM Response:")
        st.markdown(summary)
