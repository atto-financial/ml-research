# ---- Builder Stage ----
FROM python:3.11.12-slim as builder

WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ---- Final Stage ----
FROM python:3.11.12-slim

WORKDIR /app

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code and data
COPY ./app ./app
COPY ./output_data ./output_data
COPY ./plots ./plots
COPY ./save_models ./save_models
COPY ./save_scalers ./save_scalers

EXPOSE 5000

# ENV FLASK_APP=app.app
# ENV FLASK_RUN_HOST=0.0.0.0

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]