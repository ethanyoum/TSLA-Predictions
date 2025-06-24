from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd
from io import StringIO

consumer = KafkaConsumer('stock_topic',
                         bootstrap_servers='kafka:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))
producer = KafkaProducer(bootstrap_servers='kafka:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

for msg in consumer:
    data = msg.value
    df = pd.DataFrame([data])
    # Example: compute returns
    df['return'] = df['Close'].pct_change().fillna(0)
    producer.send('features_topic', value=df.to_dict(orient='records')[0])
