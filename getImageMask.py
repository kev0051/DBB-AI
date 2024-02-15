from PIL import Image
import os

# Define the input folder containing your images
input_folder = "/Users/kyvang/Documents/Capstone/data/bg_noise"

# Define the output folder for the image masks
output_folder = "/Users/kyvang/Documents/Capstone/data/bg_noise/masks"

os.makedirs(output_folder, exist_ok=True)

# Open and process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):  # Add more supported file extensions if needed
        # Open the image
        image = Image.open(os.path.join(input_folder, filename))
        
        # Create an image mask
        mask = Image.new("L", image.size)  # "L" mode for grayscale image (black and white)
        
        if image.mode == "RGBA":
            # If the image has an alpha channel, handle it
            for x in range(image.width):
                for y in range(image.height):
                    alpha = image.getpixel((x, y))[3]  # Check the alpha value (transparency)
                    mask.putpixel((x, y), 0 if alpha > 0 else 255)
        else:
            # If the image doesn't have an alpha channel, set all pixels to black
            mask = Image.new("L", image.size, 0)

        # Save the image mask to the output folder
        mask_filename = filename  
        mask.save(os.path.join(output_folder, mask_filename))

print("Image masks created successfully.")