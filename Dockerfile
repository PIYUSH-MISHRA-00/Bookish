# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files to disk and to ensure stdout/stderr is sent straight to the terminal
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies and build tools required for scikit-learn and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install cython using pip
RUN pip install cython

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "bookish_app.py"]
