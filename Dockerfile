FROM python:3

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install -r ./requirements.txt

COPY . .

ENTRYPOINT ["python"]
CMD [ "./service.py" ]
