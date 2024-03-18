# Use the official Python 3.12.0 image as the base image
FROM python:3.12.0

# Set the working directory in the container
WORKDIR /

# Copy the requirements files and install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

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