FROM python:3.12.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files and install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

# spaCy en_core_web_sm
RUN python -m spacy download en_core_web_sm

# Copy the entire application into the container
COPY . .

# Set environment variables
#ENV FLASK_APP=app.py
#ENV FLASK_RUN_HOST=0.0.0.0

# Expose port
EXPOSE 5000

# Run the application
# CMD ["flask", "run"]

ENTRYPOINT ["python", "./app.py"]
