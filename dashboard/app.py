import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/nasa_comments_with_sentiment_topic.csv")
# Map topic labels
topic_labels = {
    0: "Climate Action Urgency",
    1: "Skepticism / Hoax Claims",
    2: "Scientific Warnings",
    3: "Support for NASA",
    4: "Calls for Policy Change"
}
df["topic_label"] = df["topic"].map(topic_labels).fillna("Other")
df["date"] = pd.to_datetime(df["date"], errors='coerce')
df["year"] = df["date"].dt.year

# Streamlit config
st.set_page_config(page_title="Climate NLP Dashboard", layout="wide")
st.title(" NASA Climate Sentiment & Topic Dashboard")

# --- Section 1: Overview ---
st.markdown("""
This dashboard analyzes public climate-related comments from NASA's Facebook page using NLP techniques.
Use the filters and charts below to explore sentiment trends, topic clusters, and comment engagement.
""")

import altair as alt

# --- Section 2: Sentiment Distribution ---
st.subheader(" Sentiment Distribution")

sent_counts_df = df["sentiment_label"].value_counts().reset_index()
sent_counts_df.columns = ["Sentiment", "Count"]

bar = alt.Chart(sent_counts_df).mark_bar().encode(
    x=alt.X("Sentiment", sort="-y"),
    y="Count",
    color="Sentiment"
).properties(width=500)

st.altair_chart(bar, use_container_width=True)


# --- Section 3: Sentiment Over Time ---
st.subheader(" Sentiment Trend by Year")
sent_year = df.groupby(["year", "sentiment_label"]).size().unstack().fillna(0)
st.line_chart(sent_year)

# --- Section 4: Topic Explorer ---
st.subheader(" Topic Explorer")

topics = df["topic_label"].dropna().unique().tolist()
selected_topic = st.selectbox("Select a Topic", topics)

filtered = df[df["topic_label"] == selected_topic]

st.markdown(f"**Total Comments:** {len(filtered)}")
st.markdown(f"**Average Likes:** {filtered['likescount'].mean():.2f}")
st.markdown(f"**Average Replies:** {filtered['commentscount'].mean():.2f}")

# Show sample comments
st.markdown("###  Sample Comments")
for i, row in filtered.sample(min(3, len(filtered))).iterrows():
    st.write(f"- _{row['comment']}_\n\n**Sentiment:** {row['sentiment_label']}")

