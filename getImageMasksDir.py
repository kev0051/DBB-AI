from PIL import Image
import os

# Define the data directory containing folders with images
data_directory = "/Users/kyvang/Documents/Capstone/data"

# Get a list of subdirectories (folders) in the data directory
subdirectories = [folder for folder in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, folder))]

# Loop through each subdirectory
for subdirectory in subdirectories:
    # Define the input folder containing images
    input_folder = os.path.join(data_directory, subdirectory, 'images')

    # Define the output folder for the image masks
    output_folder = os.path.join(data_directory, subdirectory, 'masks')

    os.makedirs(output_folder, exist_ok=True)

    # Open and process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):  # Add more supported file extensions if needed
            # Open the image
            image = Image.open(os.path.join(input_folder, filename))

            # Create an image mask
            mask = Image.new("L", image.size)  # "L" mode for grayscale image (black and white)

            # Set every pixel that is not transparent to black
            for x in range(image.width):
                for y in range(image.height):
                    alpha = image.getpixel((x, y))[3]  # Check the alpha value (transparency)
                    mask.putpixel((x, y), 0 if alpha > 0 else 255)

            # Save the image mask to the output folder
            mask_filename = filename
            mask.save(os.path.join(output_folder, mask_filename))

    print(f"Image masks created successfully for folder {subdirectory}.")
