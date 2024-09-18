from PIL import Image, ImageFilter, ImageEnhance
import os
import argparse

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name of the global variable for the model')
args = parser.parse_args()


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

if __name__ == "__main__":
    directory = args.name
    convert_to_png(directory)
    process_images(directory)
