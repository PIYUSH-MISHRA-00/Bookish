import streamlit as st
import requests
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
import pickle
import numpy as np

# Function to get books from Google Books API
def get_google_books(query):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=10"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('items', [])
    else:
        return []

# Function to get books from Open Library API
def get_open_library_books(query):
    url = f"http://openlibrary.org/search.json?q={query}&limit=10"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('docs', [])
    else:
        return []

# Function to create a DataFrame from API data
def create_books_df(google_books, open_books):
    data = {
        "title": [],
        "author": [],
        "cover_image": [],
        "ratings": [],
        "user_id": []
    }

    user_ids = [f'user_{i}' for i in range(1, len(google_books) + len(open_books) + 1)]

    # Google Books data
    for i, book in enumerate(google_books):
        volume_info = book.get('volumeInfo', {})
        data["title"].append(volume_info.get('title', 'Unknown Title'))
        data["author"].append(volume_info.get('authors', ['Unknown Author'])[0])
        data["cover_image"].append(volume_info.get('imageLinks', {}).get('thumbnail', ''))
        data["ratings"].append(volume_info.get('averageRating', np.nan))  # Use NaN for missing ratings
        data["user_id"].append(user_ids[i])

    # Open Library data
    for i, book in enumerate(open_books):
        data["title"].append(book.get('title', 'Unknown Title'))
        data["author"].append(book.get('author_name', ['Unknown Author'])[0])
        cover_id = book.get('cover_i')
        cover_url = f"http://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ''
        data["cover_image"].append(cover_url)
        data["ratings"].append(np.nan)  # Use NaN for missing ratings
        data["user_id"].append(user_ids[len(google_books) + i])

    return pd.DataFrame(data)

# Function to display books in a grid
def display_books(books_df, small=False):
    cols = st.columns(5)
    for i, book in books_df.iterrows():
        with cols[i % 5]:
            if book['cover_image']:  # Check if the cover_image is not empty
                st.image(book['cover_image'], use_column_width='auto', width=80 if small else 150)
            else:
                st.text("No Image Available")  # Display a placeholder if no image
            st.write(f"**{book['title']}**")
            st.write(book['author'])

# Streamlit App
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stApp {
        background-color: #121212;
        padding: 2em;
        color: #E0E0E0;
    }
    .stHeader {
        font-size: 3em;
        color: #FFFFFF;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput input {
        background-color: #333333;
        color: #E0E0E0;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #FF6347;
        color: #E0E0E0;
        border-radius: 10px;
    }
    .stSelectbox > div {
        background-color: #333333;
        color: #E0E0E0;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='stHeader'>📚 Bookish - Your Personal Book Recommendation Platform</div>", unsafe_allow_html=True)

# Search for books
query = st.text_input("Search for books", key="book_search")
if query:
    google_books = get_google_books(query)
    open_books = get_open_library_books(query)

    if google_books or open_books:
        books_df = create_books_df(google_books, open_books)
        st.write("### Search Results")
        display_books(books_df)

        # Train the collaborative filtering model
        st.write("### Training Collaborative Filtering Model")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(books_df[['user_id', 'title', 'ratings']], reader)
        trainset, testset = train_test_split(data, test_size=0.25)

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_factors': [20, 50, 100],
            'n_epochs': [10, 20, 30],
            'biased': [True, False]
        }
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
        gs.fit(data)

        best_algo = gs.best_estimator['rmse']
        best_algo.fit(trainset)
        predictions = best_algo.test(testset)

        # Save the model to a pickle file
        with open('bookish_recommender.pkl', 'wb') as f:
            pickle.dump(best_algo, f)

        st.write("### Model Training Complete")
        st.write("The recommendation model has been trained and saved.")

        # Load the model for recommendations
        st.write("## Get Book Recommendations")
        selected_book = st.selectbox("Select a book to get recommendations", books_df['title'])
        if st.button("Recommend"):
            with open('bookish_recommender.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
            # Placeholder user ID; in a real app, gather user-specific input
            user_id = books_df['user_id'].iloc[0]
            predictions = [loaded_model.predict(user_id, book_title) for book_title in books_df['title']]
            predictions.sort(key=lambda x: x.est, reverse=True)
            
            # Display recommendations in a grid layout with smaller covers
            st.write("### Recommended Books for You")
            cols = st.columns(5)
            for prediction in predictions[:5]:  # Show top 5 recommendations
                book_title = prediction.iid
                predicted_rating = prediction.est
                book_info = books_df[books_df['title'] == book_title].iloc[0]
                with cols[predictions.index(prediction) % 5]:
                    if book_info['cover_image']:
                        st.image(book_info['cover_image'], use_column_width='auto', width=80)
                    else:
                        st.text("No Image Available")
                    st.write(f"**{book_title}**")
                    st.write(f"Predicted Rating: {predicted_rating:.2f}")
                    st.write(f"Author: {book_info['author']}")
                    st.write("---")
    else:
        st.write("No books found for your query. Try a different search term.")
