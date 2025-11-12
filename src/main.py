import os
import json
import boto3
import tarfile
from datetime import datetime
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
        logging_steps=10,
        report_to="none",
        save_strategy="no",
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

# New function: save model to tarball with date and checkpoint
def create_tarball(model_dir: str) -> str:
    # Get last checkpoint folder if exists
    checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    last_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1] if checkpoint_dirs else "final"

    date_str = datetime.now().strftime("%Y-%m-%d")
    tar_filename = os.path.join(model_dir, f"{MODEL_NAME}_{date_str}_{last_checkpoint}.tar.gz")

    with tarfile.open(tar_filename, "w:gz") as tar:
        # Only include files needed for inference
        for f in ["config.json", "model.safetensors", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]:
            path = os.path.join(model_dir, f)
            if os.path.exists(path):
                tar.add(path, arcname=f)

    print(f"[INFO] Model saved as tarball: {tar_filename}")
    return tar_filename

# Upload tarball to MinIO
def upload_model_tar_to_minio(s3_client, tar_path):
    key = os.path.basename(tar_path)
    with open(tar_path, "rb") as f:
        s3_client.put_object(Bucket=MODEL_BUCKET, Key=key, Body=f)
    print(f"[SUCCESS] Tarball uploaded to MinIO: {MODEL_BUCKET}/{key}")

def main():
    s3 = create_s3_client()
    data = download_dataset(s3)
    dataset_txt = save_dataset_txt(data)
    local_model_dir = train_gpt2(dataset_txt)
    tar_path = create_tarball(local_model_dir)
    upload_model_tar_to_minio(s3, tar_path)

if __name__ == "__main__":
    main()
