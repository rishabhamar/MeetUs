FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y portaudio19-dev python3-pyaudio

# Copy app files
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port (Render provides $PORT)
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
