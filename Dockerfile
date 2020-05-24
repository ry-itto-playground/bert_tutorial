FROM python:3.8.3-slim
USER root

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install poetry
