import os
import shutil

test_data_path = "archive/asl_alphabet_test"
output_path = "archive/asl_alphabet_test_organized"

os.makedirs(output_path, exist_ok=True)


for filename in os.listdir(test_data_path):
    if filename.endswith(".jpg"): 
        label = filename[0] 
        label_dir = os.path.join(output_path, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.move(os.path.join(test_data_path, filename), os.path.join(label_dir, filename))

print(f"Organized test dataset at: {output_path}")
