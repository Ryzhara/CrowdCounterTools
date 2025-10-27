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
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (255, 0, 255),
    "cyan": (0, 255, 255),
}

MARKER_COLOR: Final[RgbColor] = (192, 64, 192)
BLACK_COLOR: Final[RgbColor] = (0, 0, 0)
WHITE_COLOR: Final[RgbColor] = (255, 255, 255)

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


def parse_show_work(value):
    value = value.strip().lower()
    if value in ("true", "false", "grayscale"):
        return value
    raise argparse.ArgumentTypeError("show_work must be one of: true, false, grayscale")


def get_args() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count and visualize color blobs (connected components) in an image."
    )

    # Use Path type for clarity, though it will be read as str and converted back later
    parser.add_argument("image_path", type=Path, help="Path to the input image file.")

    parser.add_argument(
        "--target_rgb",
        type=parse_rgb,
        required=False,
        help="Blob color as one of [red, green, blue, yellow, cyan, magenta] or an RGB tuple like 250,120,72",
    )

    parser.add_argument(
        "--tolerance",
        type=int,
        default=30,
        help="A measure of color-match leeway for blob identification (default: 30). "
             "This is the +/- range for each RGB channel.",
    )

    parser.add_argument(
        "--show_work",
        type=parse_show_work,
        default="true",
        help="Whether to draw contours: 'true', 'false' (default), or 'grayscale' (show work on grayscale copy)",
    )

    parser.add_argument(
        "--marker_color",
        type=parse_rgb,
        default=MARKER_COLOR,
        help="Color for contour marker as name or RGB tuple (default: HOT PINK)",
    )

    args = parser.parse_args()

    # Input validation (robustness)
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Input image file not found: {args.image_path}")

    if not 0 <= args.tolerance <= 255:
        raise ValueError("Tolerance must be between 0 and 255.")

    return args


# --- Core Logic ---

#=================================
# returns list of contours found for that color within the given tolerance
def find_contours(image: np.ndarray, blob_rgb: RgbColor, tolerance: int) -> np.ndarray:
    # OpenCV uses BGR internally, so all colors must be reversed (R,G,B -> B,G,R)
    opencv_bgr = blob_rgb[::-1]

    # Calculate color ranges for cv2.inRange
    lower_bgr = np.array([max(c - tolerance, 0) for c in opencv_bgr], dtype=np.uint8)
    upper_bgr = np.array([min(c + tolerance, 255) for c in opencv_bgr], dtype=np.uint8)

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

    return contours


def count_color_blobs(
        image_path: Path,
        *,
        blob_color_list: dict[str, RgbColor],
        tolerance: int = 30,
        grayscale_copy: bool = False,
        annotation_contour_rgb: RgbColor = (192, 64, 192),
) -> Tuple[int, Any, np.ndarray, np.ndarray]:
    """
    Counts color blobs (connected components) in an image based on a target color,
    and returns the count, mask, and image with contours.

    :param image_path: Path to the input image.
    :param blob_color_list: list of RGB match (as RGB tuple).
    :param tolerance: +/- range for each RGB channel (default: 30).
    :param grayscale_copy: If True, draw contours on a grayscale copy of the image.
    :param annotation_contour_rgb: RGB color for drawing contours (default: bright green).

    :return: A tuple containing:
        - contour_list (Any): list of lists contour specifications (OpenCV format).
        - image_with_contours (np.ndarray): image with drawn contours.
    :raises FileNotFoundError: If the image cannot be read.
    """
    # Use Path.as_posix() or str() for cv2.imread
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    contour_results = {}
    for color_name, color_rgb in blob_color_list.items():
        blob_contours = find_contours(image, color_rgb, tolerance)
        contour_results[color_name] = blob_contours

    # Prepare image copy for drawing
    if grayscale_copy:
        # Convert to grayscale, then back to BGR so we can draw colored contours
        annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
    else:
        # Draw on a color copy of the original image
        annotated_image = image.copy()

    # Draw contours
    contour_bgr = annotation_contour_rgb[::-1]
    for blob_contours in contour_results.values():
        cv2.drawContours(
            annotated_image,
            blob_contours,
            contourIdx=-1,  # Draw all contours
            color=contour_bgr,
            thickness=2
        )

    return contour_results, annotated_image


def main() -> None:
    """
    Main function to execute the blob counting and visualization process.
    """
    try:
        args = get_args()

        print("--- Blob Counter Arguments ---")
        print(f"  Image Path:   {args.image_path.resolve()}")
        if args.target_rgb:
            print(f"  Special blob color:   {args.target_rgb}")
        print(f"  Show Work:    {args.show_work}")
        print(f"  Marker Color: {args.marker_color}")
        print(f"  Tolerance:    {args.tolerance}")
        print("------------------------------")

        # --- Run Core Logic ---
        blob_color_list = {f"custom {args.target_rgb}": args.target_rgb} if args.target_rgb else NAMED_COLORS
        contour_list, image_with_contours = count_color_blobs(
            image_path=args.image_path,
            blob_color_list= blob_color_list,
            tolerance=args.tolerance,
        	grayscale_copy=args.show_work == "grayscale",
            annotation_contour_rgb=MARKER_COLOR,
        )

        counts = { k: len(v) for k,v in contour_list.items() }
        counts["Total"] =sum(counts.values())

        text_summary = [ f"\t{k}: {v}" for k,v in counts.items() ]

        print(f"Found\n{"\n".join(text_summary)}\nblobs in the image.")

        # --- Output / Visualization ---
        if args.show_work in ("true", "grayscale"):
            # Construct output filename: original_name_counted.jpg
            image_dir = args.image_path.parent
            image_filename_stem = args.image_path.stem
            show_work_path = image_dir / f"{image_filename_stem}_counted_{args.show_work}.jpg"
            print(f"Writing annotation to: {show_work_path}")
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