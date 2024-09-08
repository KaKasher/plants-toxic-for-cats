import fastdup
import pandas as pd
from pathlib import Path
import os
import argparse


def remove_images(images_to_remove: list):
    for image in images_to_remove:
        if os.path.exists(image):
            os.remove(image)
        else:
            print(f"The file {image} does not exist")


def get_clusters(df, sort_by='count', min_count=2, ascending=False):
    agg_dict = {'filename': list, 'mean_distance': max, 'count': len}

    if 'label' in df.columns:
        agg_dict['label'] = list

    df = df[df['count'] >= min_count]
    grouped_df = df.groupby('component_id').agg(agg_dict)
    grouped_df = grouped_df.sort_values(by=[sort_by], ascending=ascending)
    return grouped_df


def clean_images(input_dir):
    input_dir = Path(input_dir)

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

    fd = fastdup.create(input_dir=input_dir)
    fd.run(annotations=df)

    broken_images = fd.invalid_instances()
    list_of_broken_images = broken_images['filename'].to_list()
    remove_images(list_of_broken_images)

    connected_components_df, _ = fd.connected_components()
    clusters_df = get_clusters(connected_components_df)
    cluster_images_to_keep = []
    list_of_duplicates = []

    for cluster_file_list in clusters_df.filename:
        keep = cluster_file_list[0]
        discard = cluster_file_list[1:]

        cluster_images_to_keep.append(keep)
        list_of_duplicates.extend(discard)

    remove_images(list_of_duplicates)

    outlier_df = fd.outliers()
    list_of_outliers = outlier_df['filename_outlier'].to_list()
   # remove_images(list_of_outliers)


def main():
    parser = argparse.ArgumentParser(description="Clean images in the specified directory.")
    parser.add_argument("input_dir", help="Path to the input directory containing images")
    args = parser.parse_args()

    clean_images(args.input_dir)


if __name__ == "__main__":
    main()