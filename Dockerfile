FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/app

COPY ./utils /app/utils

COPY ./lstm /app/lstm

COPY ./data_prep.json /app/app/data_prep.json