import os
import shutil
import pandas as pd

# Replace 'your_csv_file.csv' with the actual file path
csv_file_path = 'D:/capstone_project/skin_cancer/HAM10000_metadata.csv'
df = pd.read_csv(csv_file_path)

# Replace 'your_dataset_folder' with the actual path to your dataset
dataset_folder = 'D:/capstone_project/skin_cancer/pre_processed_images'

# Get unique dx_type labels from the DataFrame
dx_types = df['dx'].unique()

# Create folders for each dx_type
for dx_type_label in dx_types:
    dx_type_folder = os.path.join(dataset_folder, dx_type_label)
    os.makedirs(dx_type_folder, exist_ok=True)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    image_name = row['image_id'] + '.jpg'  # assuming images have '.jpg' extension
    source_path = os.path.join(dataset_folder, image_name)
    target_folder = os.path.join(dataset_folder, row['dx'])
    target_path = os.path.join(target_folder, image_name)

    # Move the image to the respective dx_type folder
    shutil.move(source_path, target_path)
