FROM python:3.10

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

COPY bqtest-379921-c8ee254b4e40.json /app/bqtest-379921-c8ee254b4e40.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/bqtest-379921-c8ee254b4e40.json"

EXPOSE 8080

CMD python app.py



# export GOOGLE_APPLICATION_CREDENTIALS=bqtest-379921-c8ee254b4e40.json

