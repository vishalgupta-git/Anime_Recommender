import os ,requests,pandas as pd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def download_image(data):
    url = data.image_url
    name = data['name']
    file_id = data.anime_id
    os.makedirs('./images',exist_ok =True)
    path = os.path.join('./images',str(file_id)+'.jpg')
    try:
        response = requests.get(url,timeout=10)
        if response.status_code==200:
            with open (path,'wb') as f:
                f.write(response.content)
                return name
        else:
            return ('Failed',response.status_code)
    except Exception as e:
        return "Error"
    

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def load_and_compute_sim():
    df = pd.read_csv('./datasets/cleaned.csv')
    df['title'] = df['title'].str.lower()
    df['content_features'] = df['content_features'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sim_df = pd.DataFrame(cosine_sim, index=df['title'], columns=df['title'])
    return sim_df, df

def recommend(title, sim_df, df, top_n=5):
    title = title.strip().lower()

    if title not in sim_df.columns:
        return f"Title '{title}' not found in the dataset."

    scores = sim_df[title].sort_values(ascending=False).iloc[1:top_n+1]
    recommended_titles = scores.index.tolist()

    recommendations = df[df['title'].isin(recommended_titles)][['title', 'image_url','anime_url']]
    recommendations = recommendations.set_index('title').loc[recommended_titles].reset_index()
    recommendations['score'] = recommendations['title'].map(scores)

    return recommendations
