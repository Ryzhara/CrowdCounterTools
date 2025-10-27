import cv2
import numpy as np
from pathlib import Path
import argparse

NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


def parse_rgb(value):
    """Parse RGB value: either a named color or a comma-separated tuple"""
    value = value.strip().lower()
    if value in NAMED_COLORS:
        return NAMED_COLORS[value]
    try:
        parts = tuple(int(p) for p in value.split(","))
        if len(parts) != 3 or not all(0 <= p <= 255 for p in parts):
            raise ValueError
        return parts
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Must be a named color (black, white, red, green, blue) or a comma-separated RGB tuple like 255,0,0"
        )


def parse_show_work(value):
    value = value.strip().lower()
    if value in ("true", "false", "grayscale"):
        return value
    raise argparse.ArgumentTypeError("show_work must be one of: true, false, grayscale")


def get_args():
    parser = argparse.ArgumentParser(
        description="Count and visualize RGB blobs in an image."
    )

    parser.add_argument("image_path", type=str, help="Path to the input image")

    parser.add_argument(
        "--target_rgb",
        type=parse_rgb,
        required=True,
        help="Target color as 'red', 'green', etc. or RGB tuple like 255,0,0",
    )

    parser.add_argument(
        "--tolerance",
        type=int,
        default=30,
        help="A measure of color-match leeway for blob identification (default: 30)",
    )

    parser.add_argument(
        "--show_work",
        type=parse_show_work,
        default="false",
        help="Whether to draw contours: 'true', 'false' (default), or 'grayscale'",
    )

    parser.add_argument(
        "--marker_color",
        type=parse_rgb,
        default="green",
        help="Color for contour marker as name or RGB tuple (default: green)",
    )

    return parser.parse_args()


def count_color_blobs(
    image_path,
    target_rgb,
    *,
    tolerance=30,
    grayscale_copy=False,
    contour_rgb=(0, 255, 0),
):
    """
    Counts color blobs in an image, and returns the count, mask, and image with contours.

    Parameters:
    - video_path: Path to the input image
    - target_rgb: Target color to match (as RGB tuple)
    - tolerance: +/- range for each RGB channel (default: 30)
    - grayscale_copy: If True, draw contours on a grayscale copy of the image
    - annotation_contour_rgb: RGB color for drawing contours (default: bright green)

    Returns:
    - count: number of detected blobs
    - contours: list of contour specifications
    - mask: binary mask where blobs were detected
    - image_with_contours: image with drawn contours
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    target_bgr = target_rgb[::-1]  # Convert RGB to BGR
    lower = np.array([max(c - tolerance, 0) for c in target_bgr], dtype=np.uint8)
    upper = np.array([min(c + tolerance, 255) for c in target_bgr], dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)

    # Optional: clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare image copy
    if grayscale_copy:
        image_with_contours = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_with_contours = cv2.cvtColor(image_with_contours, cv2.COLOR_GRAY2BGR)
    else:
        image_with_contours = image.copy()

    # Convert contour color to BGR
    contour_bgr = contour_rgb[::-1]
    cv2.drawContours(image_with_contours, contours, -1, contour_bgr, thickness=2)

    return len(contours), contours, mask, image_with_contours


if __name__ == "__main__":
    # python blob_counter.py path_to_image.jpg --target_rgb red --tolerance 20 --show_work true -- marker_color green

    args = get_args()
    print("Parsed arguments:")
    print(f"  image_path    = {args.image_path}")
    print(f"  target_rgb    = {args.target_rgb}")
    print(f"  show_work     = {args.show_work}")
    print(f"  marker_color  = {args.marker_color}")

    image_path = Path(args.image_path).resolve()  # get absolute path
    assert image_path.is_file()
    image_path_dir = image_path.parent
    image_path_filename = image_path.stem

    num_blobs, contours, mask, image_with_contours = count_color_blobs(
        image_path=args.image_path,
        target_rgb=args.target_rgb,
        grayscale_copy=args.show_work == "grayscale",
        contour_rgb=args.marker_color,
    )

    print(f"Found {num_blobs} {args.target_rgb} blobs in {image_path}.")

    if args.show_work in ("true", "grayscale"):
        # save the file
        show_work_path = image_path_dir / f"{image_path_filename}_counted.jpg"
        print(f"Writing {show_work_path}")
        cv2.imwrite(str(show_work_path), image_with_contours)

    print(f"Done!")
