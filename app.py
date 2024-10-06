import streamlit as st
import joblib
import pandas as pd
import altair as alt

# Load the pre-trained models
model = joblib.load('model_xgb.pkl')
vectorizer = joblib.load('countVectorizer.pkl')

def predict_sentiment(review):
    review_vector = vectorizer.transform([review])
    sentiment = model.predict(review_vector)
    return sentiment[0]

def convert_to_df(sentiment):
    sentiment_dict = {'sentiment': 'Positive' if sentiment == 1 else 'Negative'}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

def main():
    st.title("Amazon Echo Review Sentiment Analysis")
    st.subheader("Streamlit Project")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='reviewForm'):
            raw_text = st.text_area("Enter Amazon Echo Review Here")
            submit_button = st.form_submit_button(label='Analyze')

        # Layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                sentiment = predict_sentiment(raw_text)
                sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
                st.write(f'The sentiment of the review is: {sentiment_label}')

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric'
                )
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.info("Token Sentiment Analysis")
                # Token sentiment analysis could be implemented here if needed
                # For now, it’s left out since your project uses the pre-trained model

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()
