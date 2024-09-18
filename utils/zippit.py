import os
import zipfile
import argparse
import shutil

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name of the global variable for the model')
args = parser.parse_args()

def zip_files(directory, output_filename):
    zip_file = zipfile.ZipFile(output_filename, 'w')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))
    
    zip_file.close()

# Use the function
directory_to_zip = args.name
output_filename = "source.zip"

zip_files(directory_to_zip, output_filename)

shutil.move("source.zip", args.name)

