FROM python:2.7

COPY ./ /

RUN apt-get update -y
RUN apt-get install -y g++ openjdk-8-jdk python-dev python3-dev
RUN pip install -r requirements.txt

CMD ["python", "./wrapper.py"]
