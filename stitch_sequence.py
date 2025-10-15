import cv2
import argparse
from pathlib import Path
import sys

def list_files_for_stitching(directory: str, file_filter: str) -> list[str]:
    """
    Returns a sorted list of file paths in the given directory matching the glob pattern.

    :param directory: The base directory path.
    :param file_filter: A glob pattern (e.g., '*.jpg', 'image_*.png') for file names.
    :return: A list of absolute file paths (strings), sorted by name.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Use glob to find matching files
    matching_files = [
        file
        for file in dir_path.glob(file_filter)
        if file.is_file()
    ]

    # Sort files by name (as requested)
    sorted_files = sorted(matching_files, key=lambda p: p.name)

    # Convert to string paths for use with OpenCV
    return [str(file.resolve()) for file in sorted_files]


def stitch_images(image_path_list: list[str], output_filename: [Path, str]):
    """
    Stitches a list of images into a single panorama using OpenCV's Stitcher.

    :param image_path_list: A list of file paths (strings) to the input images.
    :param output_filename: The path (string or Path object) to save the output file.
    :raises RuntimeError: If image reading or stitching fails.
    """
    if not image_path_list:
        raise ValueError("The list of images to stitch is empty.")

    # 1. Read images
    images = []
    print(f"Loading {len(image_path_list)} images...")
    for path in image_path_list:
        print(f"Reading {path}...")
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image at {path}. Skipping.", file=sys.stderr)
            continue
        images.append(img)

    if len(images) < 2:
        raise RuntimeError(f"Found {len(images)} readable images. Need at least 2 images for stitching.")

    # 2. Create and run Stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA) # Use PANORAMA mode
    print("Starting image stitching...")

    # The stitch method expects a list of loaded image objects (numpy arrays), not paths
    status, stitched = stitcher.stitch(images)

    # 3. Handle result
    if status == cv2.Stitcher_OK:
        # Save the stitched image
        cv2.imwrite(str(output_filename), stitched)
        print(f"✅ Success! Panoramic image saved to: {output_filename}")
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        raise RuntimeError(f"Stitching failed: Status {status} - Need more images or could not find matching features.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        raise RuntimeError(f"Stitching failed: Status {status} - Homography estimation failed. Check image overlaps.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        raise RuntimeError(f"Stitching failed: Status {status} - Camera parameters adjustment failed.")
    else:
        raise RuntimeError(f"Stitching failed with status code {status}")


def main():
    """Main function to parse arguments and run the stitching process."""
    parser = argparse.ArgumentParser(
        description="Stitch a sequence of images into a panorama using OpenCV."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The base directory containing the images to stitch."
    )
    parser.add_argument(
        "filter",
        type=str,
        help="A file name glob pattern to filter images (e.g., '*.jpg', 'img*.png')."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="panorama.jpg",
        help="The name for the output panoramic image (default: panorama.jpg)."
    )

    args = parser.parse_args()

    try:
        # 1. Get the sorted list of image paths
        image_list_paths = list_files_for_stitching(args.directory, args.filter)
        print(f"Found {len(image_list_paths)} files matching '{args.filter}' in {args.directory}.")

        # 2. Perform stitching and saving
        stitch_images(image_list_paths, args.output)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()