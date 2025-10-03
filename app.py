import streamlit as st
import pandas as pd
from recommendation import load_and_compute_sim, recommend  

sim_df, df = load_and_compute_sim()

anime_titles = df['title'].tolist()

st.title('Anime Recommendation System')

user_input = st.text_input("Type anime name").strip().lower()

if user_input:
    suggestions = [title for title in anime_titles if user_input in title]

    if suggestions:
        selected_title = st.selectbox("Did you mean?", suggestions)

        if selected_title:
            recommendations = recommend(selected_title, sim_df, df,top_n=10)

            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                st.subheader(f"Top Recommendations for **{selected_title.title()}**:")
                for _, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <a href="{row['anime_url']}" target="_blank">
                            <img src="{row['image_url']}" width="150"/>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )

                    # Make title a clickable link
                    st.markdown(
                        f"[**{row['title'].title()}**]({row['anime_url']}) — Similarity Score: `{row['score']:.3f}`"
                    )

    else:
        st.write("No matches found.")
