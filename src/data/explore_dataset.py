import fastdup
from pathlib import Path
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Explore dataset created by fastdup. Use after clean_images.py.")
    parser.add_argument("input_dir", help="Path to the input directory containing images")
    parser.add_argument("work_dir", help="Path to the work directory that stores the outputs from cleaning the images")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)

    labels = []
    filenames = []

    class_dirs = input_dir.glob("*")
    for class_dir in class_dirs:
        if class_dir.is_dir():
            images = class_dir.glob("*")
            for image in images:
                filenames.append(str(image))
                labels.append(class_dir.name)

    df = pd.DataFrame({"filename": filenames, "label": labels})

    fd = fastdup.create(input_dir=input_dir, work_dir=work_dir)
    fd.run(annotations=df)
    fd.explore()


if __name__ == "__main__":
    main()