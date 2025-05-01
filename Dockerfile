FROM python:3.11

WORKDIR /usr/src/app

ENV VIRTUAL_ENV=/usr/src/app/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 5000
ENV FLASK_APP=app.app
CMD ["flask", "run", "--host=0.0.0.0"]