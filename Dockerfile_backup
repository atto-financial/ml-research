FROM python:3.11

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./output_data ./output_data
COPY ./plots ./plots
COPY ./save_models ./save_models
COPY ./save_scalers ./save_scalers

EXPOSE 5000
ENV FLASK_APP=app.app

CMD ["flask", "run", "--host=0.0.0.0"]