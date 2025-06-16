import cv2
from pathlib import Path

# a hacky file to stitch images that are in a directory.
# YMMV because it depends on how well the images hang together


def list_files_with_extension(directory: str, extension: str) -> list[Path]:
    """Return a list of files in the given directory with the specified extension."""
    dir_path = Path(directory)
    matching_files = [
        str(file) for file in sorted(dir_path.glob(f"*{extension}")) if file.is_file()
    ]
    return matching_files


def stitch_images(image_list: list, output_filename: [Path, str]):

    # Create a Stitcher object
    stitcher = cv2.Stitcher_create()

    # Stitch the images
    status, stitched = stitcher.stitch(image_list)
    if status == cv2.Stitcher_OK:
        # Save the stitched image
        cv2.imwrite(output_filename, stitched)
        return stitched
    else:
        raise f"Image stitching failed with status code {status}"


# next time I'll use the dang Android panoramic mode, which is built in! But in the meantime,
# let's stitch together some extracted_frames from a panoramic video, sweeping across the crowd.

# these photos don't hang together well, as they were not close enough together
# base_path = Path(r"C:\Users\ryzhara\Desktop\Democratic Party Things\Gallatin No Kings Rally Images\photos\Late Sequence")
# # Load images
# image_list = [
#     cv2.imread(base_path / 'PXL_20250614_191217901.jpg'),
#     cv2.imread(base_path / 'PXL_20250614_191220457.jpg'),
#     cv2.imread(base_path / 'PXL_20250614_191222513.jpg'),
#      cv2.imread(base_path / 'PXL_20250614_191224875.jpg'),
#     cv2.imread(base_path / 'PXL_20250614_191227065.jpg'),
#     cv2.imread(base_path / 'PXL_20250614_191229494.jpg'),
#     ]
# stitch_images(image_list, f"{base_path.stem}.jpg")

# I clipped these images and then used the windows "right click->rotate right" to orient them vertically,
# as they were taken in landscape mode. I
base_path = Path(
    r"C:\Users\ryzhara\Desktop\Democratic Party Things\Gallatin No Kings Rally Images\videos\Clips from PXL_20250614_193237148"
)

image_filename_list = list_files_with_extension(directory=base_path, extension="png")
image_list = [cv2.imread(filename) for filename in image_filename_list]


def create_groups(data: list) -> (list[list], int):
    length = len(data)
    groups = []
    # for gsize in [10, 9, 8, 7, 6, 5, 4, 3]:

    for gsize in [5, 4, 3]:
        if length % gsize != 1:  # must be 0 or >1
            return [data[i : i + gsize] for i in range(0, len(data), gsize)], gsize
    raise ValueError("Not enough data to create groups")


image_groups, group_size = create_groups(image_list)
print(f"Processing {len(image_groups)} groups")
layer = 0
image_count = len(image_list)
while image_count > 1:
    stitched_images = []
    for idx, group in enumerate(image_groups):
        print(f"Group {idx} has {len(group)} images")
        output_filename = f"{base_path.stem}_{layer}_{group_size}_{idx}.png"
        stitched_image = stitch_images(group, output_filename=output_filename)
        # weird that the returned image from stitcher is the wrong time; must reload
        stitched_images.append(cv2.imread(output_filename))
    image_count = len(stitched_images)
    image_groups = create_groups(stitched_images)
    layer += 1

# now lets stitch those together to make an intermediate
print(f"Creating frankenstein")
stitch_images(stitched_images, f"{base_path.stem}_{group_size}_frankenstein.jpg")

# print(f"Working on huge image")
# stitch_images(image_list, f"{base_path.stem}.jpg")
