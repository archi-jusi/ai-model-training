import os
import json
import boto3
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import time
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

MODEL_OUTPUT_DIR = "./gpt2_model"

# MinIO environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
DATASET_BUCKET = os.getenv("DATASET_BUCKET", "dataset")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2_finetuned")

def create_s3_client(max_retries=5, retry_delay=5):
    """
    Initialize the S3 (MinIO) client with retries if the service is not yet ready.
    """
    endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:3000")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    print(f"[INFO] Connecting to MinIO endpoint: {endpoint}")
    print("MINIO_ENDPOINT =", os.getenv("MINIO_ENDPOINT"))
    print("MINIO_ACCESS_KEY =", os.getenv("MINIO_ACCESS_KEY"))
    print("MINIO_SECRET_KEY =", "***" if os.getenv("MINIO_SECRET_KEY") else None)

    if not access_key or not secret_key:
        print("[ERROR] Missing MINIO_ACCESS_KEY or MINIO_SECRET_KEY env var.")
        return None

    for attempt in range(1, max_retries + 1):
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )

            # Quick health check
            s3.list_buckets()
            print("[SUCCESS] Successfully connected to MinIO and verified credentials.")
            return s3

        except EndpointConnectionError as e:
            print(f"[WARN] Attempt Cannot connect to MinIO endpoint: {e}")
        except (NoCredentialsError, PartialCredentialsError):
            print("[ERROR] Invalid or missing credentials for MinIO.")
            break
        except ClientError as e:
            print(f"[ERROR] MinIO client error: {e}")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error while connecting to MinIO: {e}")

        time.sleep(retry_delay)

    print("[FATAL] Failed to connect to MinIO after multiple attempts.")
    return None

def download_dataset(s3_client, dataset_file="dataset.json"):
    obj = s3_client.get_object(Bucket=DATASET_BUCKET, Key=dataset_file)
    return json.loads(obj["Body"].read().decode("utf-8"))

# Save dataset locally for HuggingFace
def save_dataset_txt(data, file_path="dataset.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(item["text"].replace("\n", " ") + "\n")
    return file_path

# Train GPT-2 on your dataset
def train_gpt2(dataset_file):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_file,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    return MODEL_OUTPUT_DIR

def upload_model_to_minio(s3_client, local_dir, model_name):
    import os
    for root, _, files in os.walk(local_dir):
        for file in files:
            file_path = os.path.join(root, file)
            s3_key = os.path.join(model_name, os.path.relpath(file_path, local_dir))
            with open(file_path, "rb") as f:
                s3_client.put_object(Bucket=MODEL_BUCKET, Key=s3_key, Body=f)
    print(f"[SUCCESS] Model uploaded to MinIO under {MODEL_BUCKET}/{model_name}")

def main():
    s3 = create_s3_client()
    data = download_dataset(s3)
    dataset_txt = save_dataset_txt(data)
    local_model_dir = train_gpt2(dataset_txt)
    upload_model_to_minio(s3, local_model_dir, MODEL_NAME)

if __name__ == "__main__":
    main()  