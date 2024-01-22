import os
from skimage import exposure, io, img_as_ubyte
from skimage.filters import gaussian

# Set the path to your dataset
dataset_path = r'D:\capstone_project\skin_cancer\training'

# Output directory for preprocessed images
output_path = r'D:\capstone_project\skin_cancer\testing'

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Function to preprocess an image
def preprocess_image(image_path, output_path):
    # Read the image
    image = io.imread(image_path)

    # Normalize pixel values to the range [0, 1]
    normalized_image = image.astype('float') / 255.0
    
    # Apply contrast enhancement using adaptive histogram equalization
    enhanced_image = exposure.equalize_adapthist(normalized_image)

    # Apply noise cancellation using Gaussian blur
    denoised_image = gaussian(enhanced_image, sigma=0.5)

    # Convert the denoised image to uint8 before saving
    denoised_image_uint8 = img_as_ubyte(denoised_image)

    # Save the preprocessed image
    output_file = os.path.join(output_path, os.path.basename(image_path))
    io.imsave(output_file, denoised_image_uint8)

# Apply preprocessing to each image in the dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(root, file)
            preprocess_image(image_path, output_path)
