import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Final, Any, Tuple, List, Union, Literal

# Define the type for an RGB color tuple
RgbColor = Tuple[int, int, int]
# Define the type for the show_work argument value
ShowWorkOption = Union[Literal['true', 'false', 'grayscale'], str]

# --- Constants and Configuration ---

# Use typing.Final for constant values
NAMED_COLORS: Final[dict[str, RgbColor]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (255, 0, 255),
    "cyan": (0, 255, 255),
}


# --- Argument Parsing Helpers ---

def parse_rgb(value: str) -> RgbColor:
    """
    Parses an RGB color value from a string.

    Accepts a named color (e.g., 'red') or a comma-separated RGB tuple (e.g., '255,0,0').

    :param value: The string to parse.
    :raises argparse.ArgumentTypeError: If the value is invalid.
    :return: An RGB color tuple (R, G, B).
    """
    value = value.strip().lower()
    if value in NAMED_COLORS:
        return NAMED_COLORS[value]

    try:
        parts = tuple(int(p) for p in value.split(","))
        # Validate length and range
        if len(parts) != 3 or not all(0 <= p <= 255 for p in parts):
            raise ValueError("RGB components must be between 0 and 255.")
        return parts  # type: ignore[return-value]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"'{value}' is invalid. Must be a named color ({', '.join(NAMED_COLORS.keys())}) "
            f"or a comma-separated RGB tuple like 255,0,0. Error: {e}"
        )


def parse_show_work(value: str) -> ShowWorkOption:
    """
    Validates the show_work argument.

    :param value: The string to validate.
    :raises argparse.ArgumentTypeError: If the value is not one of the allowed options.
    :return: The validated string.
    """
    valid_options = ("true", "false", "grayscale")
    value = value.strip().lower()
    if value in valid_options:
        return value
    raise argparse.ArgumentTypeError(
        f"show_work must be one of: {', '.join(valid_options)}"
    )


def get_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count and visualize RGB blobs (connected components) in an image."
    )

    # Use Path type for clarity, though it will be read as str and converted back later
    parser.add_argument("image_path", type=Path, help="Path to the input image file.")

    parser.add_argument(
        "--target-rgb",  # Use hyphens for command-line arguments
        type=parse_rgb,
        required=True,
        help="Target color as 'red', 'green', etc. or RGB tuple like 255,0,0.",
    )

    parser.add_argument(
        "--tolerance",
        type=int,
        default=30,
        help="A measure of color-match leeway for blob identification (default: 30). "
             "This is the +/- range for each RGB channel.",
    )

    parser.add_argument(
        "--show-work",
        type=parse_show_work,
        default="false",
        help="Whether and how to draw contours: 'true', 'false' (default), or 'grayscale'.",
    )

    parser.add_argument(
        "--marker-color",
        type=parse_rgb,
        default="green",
        help="Color for contour marker as name or RGB tuple (default: green).",
    )

    args = parser.parse_args()

    # Input validation (robustness)
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Input image file not found: {args.image_path}")

    if not 0 <= args.tolerance <= 255:
        raise ValueError("Tolerance must be between 0 and 255.")

    return args


# --- Core Logic ---

def count_color_blobs(
        image_path: Path,
        target_rgb: RgbColor,
        *,
        tolerance: int = 30,
        grayscale_copy: bool = False,
        contour_rgb: RgbColor = (0, 255, 0),
) -> Tuple[int, Any, np.ndarray, np.ndarray]:
    """
    Counts color blobs (connected components) in an image based on a target color,
    and returns the count, mask, and image with contours.

    :param image_path: Path to the input image.
    :param target_rgb: Target color to match (as RGB tuple).
    :param tolerance: +/- range for each RGB channel (default: 30).
    :param grayscale_copy: If True, draw contours on a grayscale copy of the image.
    :param contour_rgb: RGB color for drawing contours (default: bright green).

    :return: A tuple containing:
        - count (int): number of detected blobs.
        - contours (Any): list of contour specifications (OpenCV format).
        - mask (np.ndarray): binary mask where blobs were detected.
        - image_with_contours (np.ndarray): image with drawn contours.
    :raises FileNotFoundError: If the image cannot be read.
    """
    # Use Path.as_posix() or str() for cv2.imread
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    # OpenCV uses BGR internally, so all colors must be reversed (R,G,B -> B,G,R)
    target_bgr = target_rgb[::-1]

    # Calculate color ranges for cv2.inRange
    lower_bgr = np.array([max(c - tolerance, 0) for c in target_bgr], dtype=np.uint8)
    upper_bgr = np.array([min(c + tolerance, 255) for c in target_bgr], dtype=np.uint8)

    # Create the binary mask
    mask = cv2.inRange(image, lower_bgr, upper_bgr)

    # Optional: Noise reduction and gap closing (using a morphological opening)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find external contours (the boundaries of the blobs)
    # The return value of findContours can differ by OpenCV version, using _ for the hierarchy
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,  # Retrieve only external contours (the outlines of the blobs)
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Prepare image copy for drawing
    if grayscale_copy:
        # Convert to grayscale, then back to BGR so we can draw colored contours
        image_with_contours = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_with_contours = cv2.cvtColor(image_with_contours, cv2.COLOR_GRAY2BGR)
    else:
        # Draw on a color copy of the original image
        image_with_contours = image.copy()

    # Draw contours
    contour_bgr = contour_rgb[::-1]
    cv2.drawContours(
        image_with_contours,
        contours,
        contourIdx=-1,  # Draw all contours
        color=contour_bgr,
        thickness=2
    )

    return len(contours), contours, mask, image_with_contours


def main() -> None:
    """
    Main function to execute the blob counting and visualization process.
    """
    try:
        args = get_args()

        print("--- Blob Counter Arguments ---")
        print(f"  Image Path:   {args.image_path.resolve()}")
        print(f"  Target RGB:   {args.target_rgb}")
        print(f"  Tolerance:    {args.tolerance}")
        print(f"  Show Work:    {args.show_work}")
        print(f"  Marker Color: {args.marker_color}")
        print("------------------------------")

        # --- Run Core Logic ---
        num_blobs, _, _, image_with_contours = count_color_blobs(
            image_path=args.image_path,
            target_rgb=args.target_rgb,
            tolerance=args.tolerance,
            grayscale_copy=args.show_work == "grayscale",
            contour_rgb=args.marker_color,
        )

        print(f"✅ Found {num_blobs} blobs matching {args.target_rgb} in the image.")

        # --- Output / Visualization ---
        if args.show_work in ("true", "grayscale"):
            # Construct output filename: original_name_counted.jpg
            image_dir = args.image_path.parent
            image_filename_stem = args.image_path.stem
            show_work_path = image_dir / f"{image_filename_stem}_counted.jpg"

            print(f"Writing visualization to: {show_work_path}")
            # Use Path.as_posix() or str() for cv2.imwrite
            cv2.imwrite(str(show_work_path), image_with_contours)

    except (FileNotFoundError, ValueError, argparse.ArgumentTypeError) as e:
        # Catch and print specific errors raised during argument parsing or file IO
        print(f"\n❌ Error: {e}")
        # Exit with a non-zero status code to indicate failure
        exit(1)
    except Exception as e:
        # Catch all other unexpected errors
        print(f"\n❌ An unexpected error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()