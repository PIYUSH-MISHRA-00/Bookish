# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files to disc and to ensure stdout/stderr is sent straight to the terminal
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and build tools required for `scikit-surprise`
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install numpy separately
RUN pip install --upgrade pip \
    && pip install numpy

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]
CMD ["bookish_app.py"]
