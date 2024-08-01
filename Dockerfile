# Use the official lightweight Python image.
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
