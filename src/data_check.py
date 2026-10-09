import pandas as pd

import os
import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_df = pd.read_csv(os.path.join(project_path, "data", "train.csv"))
validation_df = pd.read_csv(os.path.join(project_path, "data", "validation.csv"))

print("Training shape:", train_df.shape)
print("Validation shape:", validation_df.shape)

print("\nTraining categories:")
print(train_df["label"].value_counts())

print("\nValidation categories:")
print(validation_df["label"].value_counts())

print("\nFirst 5 training images:")
print(train_df.head())