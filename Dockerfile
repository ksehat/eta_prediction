FROM python:3.8-slim

WORKDIR /app

COPY artifact /app/artifact
COPY logs /app/logs
COPY serving /app/serving

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run waitress server from the 'serving' directory
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "serving.api:app"]
