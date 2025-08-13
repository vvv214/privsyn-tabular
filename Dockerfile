# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port that the app will run on
# Cloud Run automatically sets the PORT environment variable
EXPOSE 8080

# Run the web service on container startup
# Use Gunicorn with Uvicorn workers for production
# Bind to 0.0.0.0 and the port provided by Cloud Run ($PORT)
CMD exec gunicorn web_app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 600