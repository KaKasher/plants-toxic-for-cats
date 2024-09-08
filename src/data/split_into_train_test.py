import os
import shutil
import random
import argparse

def create_train_test_split(source_folder, destination_folder, train_ratio=0.8):
    # Create the main destination folder
    os.makedirs(destination_folder, exist_ok=True)

    # Create train and test subfolders
    train_folder = os.path.join(destination_folder, 'train')
    test_folder = os.path.join(destination_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Iterate through each species folder in the source folder
    for species_folder in os.listdir(source_folder):
        species_path = os.path.join(source_folder, species_folder)

        if os.path.isdir(species_path):
            # Create corresponding species folders in train and test
            train_species_folder = os.path.join(train_folder, species_folder)
            test_species_folder = os.path.join(test_folder, species_folder)
            os.makedirs(train_species_folder, exist_ok=True)
            os.makedirs(test_species_folder, exist_ok=True)

            # Get all image files in the species folder
            image_files = [f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

            # Shuffle the image files
            random.shuffle(image_files)

            # Calculate split index
            split_index = int(len(image_files) * train_ratio)

            # Copy files to train folder
            for image_file in image_files[:split_index]:
                src = os.path.join(species_path, image_file)
                dst = os.path.join(train_species_folder, image_file)
                shutil.copy2(src, dst)

            # Copy files to test folder
            for image_file in image_files[split_index:]:
                src = os.path.join(species_path, image_file)
                dst = os.path.join(test_species_folder, image_file)
                shutil.copy2(src, dst)

            print(f"Processed {species_folder}: {split_index} train, {len(image_files) - split_index} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train-test split for image dataset.")
    parser.add_argument("source", help="Path to the source folder containing species subfolders")
    parser.add_argument("destination", help="Path to the destination folder for train-test split")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")

    args = parser.parse_args()

    create_train_test_split(args.source, args.destination, args.ratio)
    print("Train-test split completed successfully!")