# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install all Python dependencies
RUN pip install -r requirements.txt

# Run the Streamlit app
CMD ["streamlit", "run", "run_app.py", "--server.port=8501", "--server.enableCORS=false"]
