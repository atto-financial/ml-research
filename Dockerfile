FROM python:3.11

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-binary=:all: numpy==1.24.3

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 5000
ENV FLASK_APP=app.app
CMD ["flask", "run", "--host=0.0.0.0"]