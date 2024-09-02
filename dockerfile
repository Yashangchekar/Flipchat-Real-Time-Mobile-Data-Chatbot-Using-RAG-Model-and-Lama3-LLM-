# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Set environment variables
ENV LANGCHAIN_API_KEY="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"
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT="GENAIAPPWITHOPENAI"
ENV GROQ_API_KEY="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"

# Expose port 8701 for Streamlit
EXPOSE 8701

# Run the application
# CMD ["streamlit", "run", "app.py"]
CMD ["streamlit", "run", "Flipkartchatbot.py", "--server.port=8701", "--server.address=0.0.0.0"]

