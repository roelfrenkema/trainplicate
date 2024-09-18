#
# works with transformers-4.44.2
# but latest github transformers will break it
#
import argparse
import torch
import os
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline, GenerationConfig, AutoConfig

# Define command line argument parser                                                                                                                                           
parser = argparse.ArgumentParser(description='get dir')                                                                                                                   
parser.add_argument('dir', type=str, help='dir name')                                                                              
parser.add_argument('tok', type=str, help='token name')                                                                              
args = parser.parse_args()                                                                                                                                                      
                                                                                                                                                                                
# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Define the model ID
model_id = "/home/roelf/git/text-generation-webui/models/xtuner_llava-phi-3-mini-hf"

# Define the generation configuration
#generation_config = GenerationConfig(max_new_tokens=100)

# Create the pipeline
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config, "low_cpu_mem_usage": True})

# Define the prompt
prompt = "<|user|>\n<image>\nGive a description of the image.<|end|>\n<|assistant|>\n"

# Loop through all files in the directory
for filename in os.listdir(args.dir):

    # Check if the file is a PNG image
    if filename.endswith(".png"):
        # Open the image
        image = Image.open(f"{args.dir}/{filename}")
        print(f"Image: {filename}")

        # Generate text
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})

        # Get rid of empty lines
        generated_text = outputs[0]['generated_text'].replace("Give a description of the image. ", f"{args.tok}, ")
        lines = generated_text.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        result = ' '.join(non_empty_lines)

        #print(f"{result}\n")

        # Save the result to a file
        with open(os.path.splitext(f"{args.dir}/{filename}")[0] + ".txt", "w") as f:
            f.write(result)
