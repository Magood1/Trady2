# Dockerfile
# Use a slim image to reduce size, which is based on Debian
FROM python:3.11-slim

# Set environment variables to prevent interactive prompts
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for building Python packages like LightGBM
# build-essential, cmake are needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Default command to run the application (can be overridden in docker-compose.yml)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]