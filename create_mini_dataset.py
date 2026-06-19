import os
import shutil
import pandas as pd

def create_mini_dataset():
    base_dir = r"c:\VLTCRISIS"
    source_data_dir = os.path.join(base_dir, "Data", "crisismmd")
    source_img_root = os.path.join(base_dir, "CrisisMMD_v2.0", "CrisisMMD_v2.0")
    
    out_dir = os.path.join(base_dir, "MiniDataset")
    out_csv_dir = os.path.join(out_dir, "csv")
    out_img_dir = os.path.join(out_dir, "images")
    
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    
    splits = {
        "train.csv": 500,
        "dev.csv": 100,
        "test.csv": 100
    }
    
    copied_images = 0
    missing_images = 0
    
    for filename, row_limit in splits.items():
        csv_path = os.path.join(source_data_dir, filename)
        if not os.path.exists(csv_path):
            print(f"Could not find {csv_path}")
            continue
            
        print(f"Processing {filename}...")
        df = pd.read_csv(csv_path)
        
        # Take a balanced sample of N rows total (N/5 per class)
        if 'label' in df.columns:
            n_per_class = row_limit // 5
            # Group by label and sample up to n_per_class (or all if fewer exist)
            df_mini = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n_per_class), random_state=42)
            ).copy()
            # If we didn't reach row_limit, we could sample more from larger classes,
            # but a perfectly balanced dataset is better for training a small model!
        else:
            df_mini = df.head(row_limit).copy()
        
        # Save new CSV
        out_csv_path = os.path.join(out_csv_dir, filename)
        df_mini.to_csv(out_csv_path, index=False)
        
        # Copy the images referenced in this mini CSV
        for img_rel_path in df_mini['image_path']:
            src_img_path = os.path.join(source_img_root, img_rel_path)
            
            # The UI will expect the images to be directly in the image directory or maintain structure?
            # Let's maintain the folder structure inside our out_img_dir so the relative paths still work
            dest_img_path = os.path.join(out_img_dir, img_rel_path)
            
            # Create subdirectories if they don't exist
            os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
            
            if os.path.exists(src_img_path):
                if not os.path.exists(dest_img_path):
                    shutil.copy2(src_img_path, dest_img_path)
                    copied_images += 1
            else:
                missing_images += 1
                
    print("\n--- Done! ---")
    print(f"Successfully copied {copied_images} images.")
    if missing_images > 0:
        print(f"Warning: Could not find {missing_images} source images.")
    print(f"Your tiny dataset is ready at: {out_dir}")

if __name__ == '__main__':
    create_mini_dataset()
