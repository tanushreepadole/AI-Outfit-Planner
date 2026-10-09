import pandas as pd

train_df = pd.read_csv(
    r"C:\Users\ASUS VIVOBOOK 15\OneDrive\Attachments\python crash\AI-Outfit-Planner\data\train.csv"
)

print(train_df.head())
print("Total training images:", len(train_df))
from PIL import Image
import numpy as np

# Take the first image path from CSV
image_path = train_df.loc[0, "image_path"]

print("Image path:", image_path)

# Open image
image = Image.open(image_path)

# Resize
image = image.resize((224, 224))

# Convert to NumPy array
image_array = np.array(image)

# Normalize
image_array = image_array / 255.0

print("Image shape:", image_array.shape)
print("Minimum:", image_array.min())
print("Maximum:", image_array.max())

from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Step 1: Open the image
    image = Image.open(image_path).convert("RGB")

    # Step 2: Resize to a fixed size
    image = image.resize((224, 224))

    # Step 3: Convert image into a NumPy array
    image_array = np.array(image)

    # Step 4: Normalize pixel values from 0–255 to 0–1
    image_array = image_array / 255.0

    return image_array

if __name__ == "__main__":
    image_path = r"D:\Telegram\Re-TypeAware\top\100025247.jpg"

    processed_image = preprocess_image(image_path)

    print("Image shape:", processed_image.shape)
    print("Minimum pixel value:", processed_image.min())
    print("Maximum pixel value:", processed_image.max())
