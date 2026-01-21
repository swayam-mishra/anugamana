# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all backend files
COPY . .

# Create the user FIRST so we can assign permissions to them
RUN useradd -m -u 1000 user

# Create directory and generate the Vector Index (runs as root)
RUN mkdir -p chroma_gita
RUN python indexer.py

# --- THE FIX ---
# Give the 'user' full ownership of the database folder and ALL its contents
RUN chown -R user:user chroma_gita
# ----------------

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860
EXPOSE 7860

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]