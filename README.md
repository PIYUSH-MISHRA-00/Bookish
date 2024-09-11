![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

# 📚 Bookish - Your Personal Book Recommendation Platform

**Bookish** is a web application that provides personalized book recommendations using collaborative filtering. It integrates data from Google Books and Open Library APIs to deliver a seamless book search and recommendation experience.

## Features

- **Book Search**: Search for books using Google Books and Open Library APIs.
- **Book Display**: View search results with book titles, authors, cover images, and ratings.
- **Recommendation Engine**: Get personalized book recommendations based on collaborative filtering.
- **Model Training**: The application trains a recommendation model using user ratings and book data.

## Technologies Used

- **Streamlit**: For creating the web application.
- **Requests**: For making HTTP requests to book APIs.
- **Pandas**: For data manipulation.
- **Surprise**: For collaborative filtering and model training.
- **Pickle**: For saving and loading the recommendation model.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/bookish.git
   cd bookish
   ```
    Install the required packages:

    ```

    pip install -r requirements.txt
    ```

## Usage

    Run the Streamlit app:

    ```
    streamlit run app.py
    ```
## Search for books:
Enter a search query in the text input field.
View the search results displayed in a grid format.

## Get Book Recommendations:
Select a book from the dropdown menu.
Click the "Recommend" button to receive personalized recommendations.