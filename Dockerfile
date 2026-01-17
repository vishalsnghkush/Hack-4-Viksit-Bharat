# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# System dependencies removed as we use opencv-python-headless

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Run Streamlit when the container launches
# server.port must match EXPOSE
CMD ["streamlit", "run", "web_app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
