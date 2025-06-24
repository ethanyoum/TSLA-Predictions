from flask import Flask, request
from kafka import KafkaProducer
import json

app = Flask(__name__)
producer = KafkaProducer(bootstrap_servers='kafka:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    producer.send('stock_topic', value=data)
    return 'Data sent to Kafka', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
