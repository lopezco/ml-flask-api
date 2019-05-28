FROM python:3

WORKDIR /usr/src/app

COPY . .
RUN pip install -r ./requirements-service.txt
RUN pip install -r ./requirements.txt

ENTRYPOINT ["python"]
CMD [ "./service.py" ]
