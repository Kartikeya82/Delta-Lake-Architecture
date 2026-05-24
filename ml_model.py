from kafka import KafkaProducer
import pandas as pd
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

file = "training.1600000.processed.noemoticon.csv"

df = pd.read_csv(file, encoding="latin-1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]

print("🚀 Streaming started...")

count = 0

for _, row in df.iterrows():
    producer.send("twitter_stream", row.to_dict())
    count += 1

    if count % 1000 == 0:
        print(f"Sent {count} records")

    time.sleep(0.01)  # simulate streaming

producer.flush()
