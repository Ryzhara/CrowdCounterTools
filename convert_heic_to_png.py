import os
from PIL import Image
import pillow_heif

# Register the HEIF opener plugin for Pillow
pillow_heif.register_heif_opener()


def convert_heic_to_png(directory_path):
    """
    Enumerates all HEIC images in a given directory and converts each one
    to a PNG version, saving the new file in the same directory.

    Args:
        directory_path (str): The path to the directory containing the images.
    """
    # Ensure the provided path is a valid directory
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    print(f"Scanning directory: '{directory_path}' for HEIC images...")

    heic_count = 0
    converted_count = 0

    # Walk through the directory (including subdirectories)
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            # Check for common HEIC/HEIF extensions (case-insensitive)
            if file_name.lower().endswith((".heic", ".heif")):
                heic_count += 1
                heic_path = os.path.join(root, file_name)

                # Construct the new PNG file path
                base_name = os.path.splitext(file_name)[0]
                png_file_name = f"{base_name}.png"
                png_path = os.path.join(root, png_file_name)

                print(f"  - Converting: '{file_name}' -> '{png_file_name}'")

                try:
                    # Open the HEIC image using Pillow (with the registered plugin)
                    image = Image.open(heic_path)

                    # Save the image as PNG
                    image.save(png_path, format="png")
                    converted_count += 1
                    print(f"    ✅ Success: Saved to '{png_path}'")

                except Exception as e:
                    print(f"    ❌ Error converting '{file_name}': {e}")

    # Summary
    print("\n--- Conversion Summary ---")
    print(f"Total HEIC files found: {heic_count}")
    print(f"Total files successfully converted to PNG: {converted_count}")
    if heic_count != converted_count:
        print("Note: Some files may have failed to convert. Check the logs for errors.")


# --- Script Execution ---
if __name__ == "__main__":
    # IMPORTANT: Replace 'YOUR_DIRECTORY_PATH_HERE' with the actual directory path.
    # For example: directory_to_scan = "/Users/username/Pictures/HEIC_Photos"
    directory_to_scan = r"C:\Users\ryzhara\Desktop\Indivisible\NK2\Photos\Nashville\Felipe"  # 🚨 Replace this placeholder!

    # Check if the placeholder was replaced
    if directory_to_scan == "YOUR_DIRECTORY_PATH_HERE":
        print("\n*** ACTION REQUIRED ***")
        print("Please edit the script and replace 'YOUR_DIRECTORY_PATH_HERE' ")
        print("with the actual path of the directory you want to process.")
        print("Exiting script.")
    else:
        convert_heic_to_png(directory_to_scan)
