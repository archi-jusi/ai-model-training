import os
import json
import boto3
import tarfile
import time
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

MODEL_OUTPUT_DIR = "./model_output"

# Environment Variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
DATASET_BUCKET = os.getenv("DATASET_BUCKET", "dataset")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2_finetuned")

# -------------------------
# MinIO Client
# -------------------------
def create_s3_client(max_retries=5, retry_delay=5):

    endpoint = MINIO_ENDPOINT or "http://localhost:3000"

    print(f"[INFO] Connecting to MinIO at {endpoint}")
    print("MINIO_ACCESS_KEY =", MINIO_ACCESS_KEY)
    print("MINIO_SECRET_KEY =", "***")

    for attempt in range(max_retries):
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
            )
            s3.list_buckets()
            print("[SUCCESS] Connected to MinIO!")
            return s3
        except Exception as e:
            print(f"[WARN] MinIO not ready ({attempt+1}/{max_retries}): {e}")
            time.sleep(retry_delay)

    print("[FATAL] Failed to connect to MinIO.")
    return None


# -------------------------
# Download dataset
# -------------------------
def download_dataset(s3_client, dataset_file="dataset.json"):
    print("[INFO] Downloading dataset from MinIO...")
    obj = s3_client.get_object(Bucket=DATASET_BUCKET, Key=dataset_file)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    print(f"[INFO] Loaded {len(data)} book items.")
    return data


# -------------------------
# Preprocessing with sliding window
# -------------------------
def preprocess_dataset(data, tokenizer, block_size=256, stride=128):
    print(f"[INFO] Preprocessing with block_size={block_size} stride={stride}")

    full_text = "\n\n".join([item["text"] for item in data])
    print(f"[INFO] Combined dataset length: {len(full_text)} chars")

    tokenized = tokenizer(full_text, return_attention_mask=False)
    input_ids = tokenized["input_ids"]
    print(f"[INFO] Total tokens: {len(input_ids)}")

    chunks = []
    for i in range(0, len(input_ids) - block_size, stride):
        chunk = input_ids[i : i + block_size]
        chunks.append({"input_ids": chunk, "labels": chunk.copy()})

    print(f"[INFO] Created {len(chunks)} chunks")
    return Dataset.from_list(chunks)

# -------------------------
# Training
# -------------------------
def train_model(data):

    print("[INFO] Loading model distilgpt2 (best for Mac CPU)")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = preprocess_dataset(
        data=data,
        tokenizer=tokenizer,
        block_size=256,
        stride=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=20,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        fp16=False,         # CPU only
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("[INFO] Starting training…")
    trainer.train()

    print("[INFO] Saving model...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    return MODEL_OUTPUT_DIR

# -------------------------
# Tarball Creation
# -------------------------
def create_tarball(model_dir: str) -> str:

    date_str = datetime.now().strftime("%Y-%m-%d")
    tar_filename = os.path.join(model_dir, f"{MODEL_NAME}_{date_str}.tar.gz")

    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "merges.txt",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    with tarfile.open(tar_filename, "w:gz") as tar:
        for f in required:
            path = os.path.join(model_dir, f)
            if os.path.exists(path):
                tar.add(path, arcname=f)
            else:
                print(f"[WARN] Missing {f}")

    print(f"[INFO] Created model tarball: {tar_filename}")
    return tar_filename


# -------------------------
# Upload to MinIO
# -------------------------
def upload_model_tar(s3_client, tar_path):
    key = os.path.basename(tar_path)
    s3_client.upload_file(tar_path, MODEL_BUCKET, key)
    print(f"[SUCCESS] Uploaded model to MinIO: {MODEL_BUCKET}/{key}")


# -------------------------
# Main
# -------------------------
def main():
    s3 = create_s3_client()
    if not s3:
        return

    data = download_dataset(s3)
    model_dir = train_model(data)
    tar_path = create_tarball(model_dir)
    upload_model_tar(s3, tar_path)


if __name__ == "__main__":
    main()
