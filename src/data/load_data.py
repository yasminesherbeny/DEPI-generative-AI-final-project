import kagglehub
import os
import pandas as pd

def load_dataset():
    path = kagglehub.dataset_download("mewbius/ecommerce-products")
    csv_path = os.path.join(path, "productsfull2.csv")

    df = pd.read_csv(csv_path)
    print(f"Dataset loaded ✓ | Shape: {df.shape}")

    return df