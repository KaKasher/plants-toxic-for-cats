import argparse
from pathlib import Path
from bing_image_downloader import downloader
from tqdm import tqdm


def load_plants(filename):
    with open(filename, "r") as f:
        plants = f.read().splitlines()

    dict_plants = {}
    for line in plants:
        parts = line.split("(")
        plant_name = parts[0].strip()
        aliases = parts[1].rstrip(')').strip() if len(parts) > 1 else []
        aliases = [s.strip() for s in aliases.split(",")] if aliases else []

        if plant_name not in dict_plants:
            dict_plants[plant_name] = aliases

    return dict_plants


def download_images(dict_plants, limit, output_path):
    for plant_name, aliases in tqdm(dict_plants.items()):
        for plant in [plant_name] + aliases:
            print(f"Downloading images for {plant}")
            downloader.download(
                plant,
                limit=limit,
                output_dir=output_path / plant_name,
                adult_filter_off=False,
                force_replace=False,
                timeout=60,
                verbose=False
            )


def main():
    parser = argparse.ArgumentParser(description="Download plant images from Bing.")
    parser.add_argument("filename", type=str, help="Path to the input file containing plant names")
    parser.add_argument("limit", type=int, help="Maximum number of images to download per plant")
    parser.add_argument("output_path", type=str, help="Path to the output directory for downloaded images")

    args = parser.parse_args()

    filename = Path(args.filename)
    output_path = Path(args.output_path)

    if not filename.exists():
        print(f"File '{filename}' not found")
        return

    dict_plants = load_plants(filename)
    download_images(dict_plants, args.limit, output_path)


if __name__ == "__main__":
    main()