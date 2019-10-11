FROM python:3

WORKDIR /usr/src/app

# Copy and install service requirements
COPY requirements-service.txt .
RUN pip install -r ./requirements-service.txt

# Copy model requirements
COPY requirements.txt .
RUN pip install -r ./requirements.txt

# Copy code and model
COPY . .

ENTRYPOINT ["python"]
CMD [ "./service.py" ]
