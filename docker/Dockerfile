FROM python:3

WORKDIR /app

RUN pip install --upgrade pip

# Copy and install service requirements
COPY requirements-service.txt .
RUN pip install -r ./requirements-service.txt

# Copy model requirements
COPY requirements.txt .
RUN pip install -r ./requirements.txt

# Copy code and model
COPY . .

EXPOSE 5000

CMD gunicorn -b 0.0.0.0:5000 service --timeout 300 --workers=2 --threads=4 --worker-class=gthread

