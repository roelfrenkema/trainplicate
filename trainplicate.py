#
# Trainplicate a script by Roelf Renkema.
#
# This program is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# v1.0.10
#
# Updates:
# - Changed the production of captions to remove linefeeds.
#
import os
import torch
import argparse
import zipfile
import shutil
import replicate
import datetime
from replicate import models
from replicate.exceptions import ReplicateError
from PIL import Image, ImageFilter, ImageEnhance
from transformers import BitsAndBytesConfig, pipeline
from termcolor import colored

########################################################################

# This should reflect you Replicate, Github and Huggingface username
owner="READ THE COMMENT"

# This should be the full path to the directory the model files are in
mpath="READ THE COMMENT"

########################################################################

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name/path of the lora')
parser.add_argument('tok', type=str, help='token name')                                                                              
parser.add_argument('description', type=str, help='desciption of the lora')                                                                              
args = parser.parse_args()

# Step 1 convert all images to PNG and remove anything not png
def convert_to_png(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            with Image.open(file_path) as img:
                new_filename = os.path.splitext(filename)[0] + '.png'
                new_file_path = os.path.join(directory, new_filename)
                img.save(new_file_path, 'PNG')
        except IOError:
            print(f"Cannot convert {file_path}")

    for filename in os.listdir(directory):
        # Create the full file path
        file_path = os.path.join(directory, filename)
    
        # Check if it is a file and does not end with .png
        if os.path.isfile(file_path) and not filename.endswith('.png'):
            # Delete the file
            os.remove(file_path)
            print(f'Deleted: {file_path}')


# Step 2 resize all images to 1024 box
# Routine will resize all images to a 1024 box keeping aspect ratio
# You can change "max_size" for other formats.
def process_images(directory):
    max_size = 1024
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}")

            with Image.open(file_path) as img:
                # Remove metadata
                img = img.copy()

                # Unsharp masking
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

                # Calculate new dimensions to maintain aspect ratio
                width, height = img.size
                if width > height:
                    new_width = max_size
                    new_height = int((max_size / width) * height)
                else:
                    new_height = max_size
                    new_width = int((max_size / height) * width)

                # Resize the image
                img = img.resize((new_width, new_height), Image.HAMMING)

                # Save the processed image with optimized settings and without metadata
                img.save(file_path, 'PNG', optimize=True)


# Step 3 caption images
# Caption all png files in the directory
# You can change the model, but then you probably have to change the 
# prompt too.
# You can change the prompt, but you will then have to change the output
# replacement too. 
def create_captions(directory, token, model):

    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Define the model ID
    model_id = model

    # Create the pipeline
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config, "low_cpu_mem_usage": True})

    # Define the prompt
    # You can change the prompt, but you will then have to change 
    # the output replacement too. 
    prompt = "<|user|>\n<image>\nGive a description of the image.<|end|>\n<|assistant|>\n"

    # Loop through all files in the directory
    for filename in os.listdir(directory):

        # Check if the file is a PNG image
        if filename.endswith(".png"):
            # Open the image
            image = Image.open(f"{directory}/{filename}")
            print(f"Image: {filename} being captioned.")

        # Generate text
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})

        # Get rid of empty lines and replace the prompt by our token.
        generated_text = outputs[0]['generated_text'].replace("Give a description of the image. ", f"{token}, ")
        # Replace newlines with spaces and remove extra spaces
        result = ' '.join(generated_text.split())

        # Save the result to a file
        with open(os.path.splitext(f"{directory}/{filename}")[0] + ".txt", "w") as f:
            f.write(result)


# Step 4 zip images and captions(= *.txt)
# There should not be anything else in your directory         
def zip_files(directory, output_filename):
    zip_file = zipfile.ZipFile(output_filename, 'w')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))
    
    zip_file.close()


# Step 5 is a one liner in the main routine


# Step 6 create a model if one does not exist
# Beware if it exists it will use that for training
def create_model(directory,user,description):
    try:
        # Try to create a new model
        model = models.create(
            name=f"{directory}",
            owner=f"{user}",
            visibility="public",  # or "private" if you prefer
            hardware="gpu-t4",  # Replicate will override this for fine-tuned models
            description=f"{description}"
        )
        print(f"Model created: {directory}")
    except ReplicateError as e:
        if f'A model with that name and owner already exists.' in str(e):
            print("Model already exists, using it for training.")
        else:
            raise  # If this is not a "model already exists" error, re-raise the exception.

# Step 7 send it of to Replicate to develop
# Be sure your replicate token is set in your enveronment.
# If you use Huggingface a write key should be available in your environment as INFERENCE_WRITE.
def train_model(directory,user,token):
    hugkey=os.getenv("INFERENCE_WRITE")
    training = replicate.trainings.create(
        version="ostris/flux-dev-lora-trainer:7f53f82066bcdfb1c549245a624019c26ca6e3c8034235cd4826425b61e77bec",
        input={
            "input_images": open(f"{directory}/source.zip", "rb"),
            "steps": 1000,
            "lora_rank": 16,
            "optimizer": "adamw8bit",
            "batch_size": 1,
            "resolution": "512,768,1024",
            "autocaption": False,
            "trigger_word": f"{token}",
            "learning_rate": 0.0004,
            "hf_token": f"{hugkey}",  # optional
            "hf_repo_id": f"{user}/{directory}",  # optional
        },
    destination=f"{user}/{directory}"
    )

    print(f"Training started: {training.status}")
    print(f"Training URL: https://replicate.com/p/{training.id}")


# A little helper routine that prints a text with a timestamp
# Great to follow execution and time.
def time_stamp(text):
    current_time = datetime.datetime.now()
    time_text = current_time.strftime("%H:%M:%S")
    print(colored(f"{time_text} {text} ",'white', 'on_light_red'))
        
if __name__ == "__main__":
    
    # Step 1 convert all images to PNG
    time_stamp("Step 1 convert all images to PNG")
    convert_to_png(args.name)
    # Step 2 resize all images to 1024 box
    time_stamp("Step 2 resize all images to 1024 box")
    process_images(args.name)
    # Step 3 caption images
    time_stamp("Step 3 caption images")
    create_captions(args.name,args.tok,mpath)
    # Step 4 zip images and captions
    time_stamp("Step 4 zip images and captions")
    zip_files(args.name, "source.zip")
    # Step 5 move the zip into the lora dir
    time_stamp("Step 5 move the zip into the lora dir")
    shutil.move("source.zip", args.name)
    exit()
    # Step 6 create a model if one does not exist
    time_stamp("Step 6 create a model if one does not exist")
    create_model(args.name,owner,args.description)
    # Step 7 send it of to Replicate to bake
    time_stamp("Step 7 send it of to Replicate to bake")
    train_model(args.name,owner,args.tok)
    time_stamp("And another beautifull LoRa is baking in the oven!")
