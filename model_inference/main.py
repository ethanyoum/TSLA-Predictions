from kafka import KafkaConsumer
import json
import pandas as pd
import torch
from five_day_return import GRUWithAttentionClassifier
import os

## Model Loading
INPUT_SIZE = 15
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.05
BIDIRECTIONAL = False

# Load the model structure and the trained weights
model = GRUWithAttentionClassifier(
    input_size = INPUT_SIZE,
    hidden_size = HIDDEN_SIZE,
    num_layers = NUM_LAYERS,
    dropout = DROPOUT,
    bidirectional = BIDIRECTIONAL
)
model.load_state_dict(torch.load('tsla_attention.pt', map_location=torch.device('cpu')))
model.eval()

## Kafka Consumer
consumer = KafkaConsumer('features_topic',
                         bootstrap_servers='kafka:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

print("Model loaded. Waiting for messages...")

output_file = "predictions.csv"
header_written = os.path.exists(output_file)

for msg in consumer:
    features = msg.value
    df = pd.DataFrame([features])
    
    ## Inference
    # 1. Prepare data for the model (assumes all columns except target are features)
    feature_columns = [col for col in df.columns if col not in ['target', 'timestamp']] # Example
    input_data = df[feature_columns].values
    
    # 2. Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0) # Add batch dimension

    # 3. Get prediction
    with torch.no_grad():
        prediction_tensor = model(input_tensor)
        prediction = (prediction_tensor.item() > 0.5) * 1 # Convert to 0 or 1
    
    result = df.copy()
    result['prediction'] = prediction

    # Save to CSV
    result.to_csv(output_file, mode='a', header=not header_written, index=False)
    header_written = True

    print(f"Prediction: {prediction}, Features: {df.to_dict()}")
