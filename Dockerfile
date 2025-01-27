# Use the official PyTorch image as a base
#FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Define the entry point for the container
ENTRYPOINT ["python", "src/train2.py"]

