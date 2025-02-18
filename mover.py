import os
import shutil
import random

def move_images(train_dir, test_dir, num_images=50):
    for letter_folder in os.listdir(train_dir):
        train_letter_path = os.path.join(train_dir, letter_folder)
        test_letter_path = os.path.join(test_dir, letter_folder)

        if os.path.isdir(train_letter_path):
            if not os.path.exists(test_letter_path):
                os.makedirs(test_letter_path)

            files = [f for f in os.listdir(train_letter_path) if os.path.isfile(os.path.join(train_letter_path, f)) and f.lower().endswith('.jpg')]

            # Randomly select files to move
            files_to_move = random.sample(files, min(num_images, len(files)))

            for file_name in files_to_move:
                src = os.path.join(train_letter_path, file_name)
                dst = os.path.join(test_letter_path, file_name)
                shutil.move(src, dst)

            print(f"Moved {len(files_to_move)} images from {train_letter_path} to {test_letter_path}.")

if __name__ == "__main__":
    train_directory = "archive/asl_alphabet_train"
    test_directory = "archive/asl_alphabet_test"
    
    move_images(train_directory, test_directory, num_images=450)
