import replicate
import argparse
from replicate import models
from replicate.exceptions import ReplicateError

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name of the global variable for the model')
parser.add_argument('token', type=str, help='The name of the token')
parser.add_argument('description', type=str, help='The description')
args = parser.parse_args()

# Assign command line arguments to global variables
owner = "roelfrenkema"

try:
    # Try to create a new model
    model = models.create(
        name=f"{args.name}",
        owner=f"{owner}",
        visibility="public",  # or "private" if you prefer
        hardware="gpu-t4",  # Replicate will override this for fine-tuned models
        description=f"{args.description}"
    )
    print(f"Model created: {args.name}")
except ReplicateError as e:
    if f'A model with that name and owner already exists.' in str(e):
        print("Model already exists, using it for training.")
    else:
        raise  # If this is not a "model already exists" error, re-raise the exception.
	
# Now use this model as the destination for your training

training = replicate.trainings.create(
    version="ostris/flux-dev-lora-trainer:7f53f82066bcdfb1c549245a624019c26ca6e3c8034235cd4826425b61e77bec",
    input={
        "input_images": open(f"{args.name}/source.zip", "rb"),
        "steps": 1000,
        "lora_rank": 16,
        "optimizer": "adamw8bit",
        "batch_size": 1,
        "resolution": "512,768,1024",
        "autocaption": False,
        "trigger_word": f"{args.token}",
        "learning_rate": 0.0004,
        "hf_token": "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",  # optional
        "hf_repo_id": f"{owner}/{args.name}",  # optional
    },
    destination=f"{owner}/{args.name}"
)

print(f"Training started: {training.status}")
print(f"Training URL: https://replicate.com/p/{training.id}")

