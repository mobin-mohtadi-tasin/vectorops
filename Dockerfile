# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency definition first for caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend codebase and models
COPY backend /app/backend

# Copy frontend folder so FastAPI can serve the dashboard
COPY frontend /app/frontend

# Set environment variable for port (Render/Railway dynamically inject PORT)
ENV PORT=8000
EXPOSE 8000

# Run uvicorn from the backend workspace directory
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
