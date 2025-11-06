import os
import io
import json
import boto3
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from botocore.exceptions import ClientError

# MinIO environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
DATASET_BUCKET = os.getenv("DATASET_BUCKET", "dataset")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2_finetuned")

# Connect to MinIO
def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

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
        output_dir="./gpt2_model",
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
    trainer.save_model("./gpt2_model")
    return "./gpt2_model"

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