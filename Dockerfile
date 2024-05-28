# Use the official Python image as base
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Expose port 8080
EXPOSE 8080

# Command to run your application

CMD ["python", "your_script.py"]
