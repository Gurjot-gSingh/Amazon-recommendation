import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')
df = df.dropna(subset=['product_name', 'category', 'rating', 'review_content', 'review_summary'])
df['review_content'] = df['review_content'].str.lower()

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['review_content'])

def recommend_product(user_input, top_n=5):
    # Transform the user input
    user_input_tfidf = vectorizer.transform([user_input])
    
    # Calculate cosine similarity between user input and product descriptions
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()
    
    # Get the top n product indices
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    
    # Get the top n product details
    recommended_products = df.iloc[top_indices]
    
    return recommended_products

st.title("Amazon Product Recommendation")

# Input prompt
prompt = st.text_input("Enter your product search query:")

if prompt:
    # Generate recommendations
    recommendations = recommend_product(prompt)
    st.write("Recommended Products:")
    for idx, row in recommendations.iterrows():
        st.write(f"**Product Name:** {row['product_name']}")
        st.write(f"**Category:** {row['category']}")
        st.write(f"**Discounted Price (USD):** {row['discounted_price']}")
        st.write(f"**Actual Price (USD):** ₹{row['actual_price']:.2f}")
        st.write(f"**Rating:** {row['rating']}")
        st.write(f"**Review Sentiment:** {row['review_sentiment']}")
        st.write(f"**Review Summary:** {row['review_summary']}")
        st.write(f"**Product Link :** [Link]({row['product_link']})")
        st.write("---")