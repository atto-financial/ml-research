FROM python:3.11-slim

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 5000

ENV FLASK_APP=app.app

CMD ["flask", "run", "--host=0.0.0.0"]