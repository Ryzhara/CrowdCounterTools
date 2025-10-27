# CrowdCounterTools
Simple tools to count crowd density at public gatherings

Creating a panoramic image allows using your favorite video editor and human vision, or automation
to tag stuff in your panoramic image with 'blobs' of paint, whether from the 
video extraction or other source, that need to be counted with a known color.
Say, RED=(255,0,0). Then you can use the blob counter to count the markers.

## Blob Counter Examples
```commandline
# To take defaults, so count red,green,blue,yellow,cyan, and magenta and show result on grayscale copy:
python blob_counter.py  path_to_image.png

# To count just red (255,0,0) blobs and show work on original color copy
python blob_counter.py --target_rgb red --show_work true path_to_image.ext

# To count custom color blobs with color, tolerance of 20, skip making a copy, mark the counted blobs with green
python blob_counter.py --target_rgb 240,30,30 --tolerance 20 --show_work false --marker_color green  path_to_image.ext
```
The blob counter tool works on whatever OpenCV can read. It has been tested on both PNG and JPG images.

## Blob Counter Syntax
```commandline
usage: blob_counter.py [-h] 
        [--target_rgb TARGET_RGB]
        [--tolerance TOLERANCE] 
        [--show_work SHOW_WORK] 
        [--marker_color MARKER_COLOR] 
        image_path

Count and visualize RGB blobs in an image.

positional arguments:
  image_path    Path to the input image

options:
  -h, --help
        show this help message and exit
  --target_rgb TARGET_RGB
        Blob color as one of [red, green, blue, yellow, cyan, magenta] or an RGB tuple like 250,120,72; default is all r,g,b,y,c,m
  --tolerance TOLERANCE
        A measure of color-match leeway for blob identification (default: 30). This is the +/- range for each RGB channel.
  --show_work SHOW_WORK
        Whether to draw contours: true (default), false, or grayscale (show work on grayscale copy)
  --marker_color MARKER_COLOR
        Color for contour marker as name or RGB tuple (default: HOT PINK)
```


## Video-to-Panorama Syntax
This tool has been tested on MP4 images. It tries to create a panorama by extracting frames
from the video file with spacing you specify.

```commandline
usage: video_to_panorama.py [-h] 
        [--start_time START_TIME] 
        [--end_time END_TIME] 
        [--extract_fps EXTRACT_FPS] 
        [--rotate_frames {cw90,ccw90,180}]
        video_path

Extract frames from a video to create a panoramic image.

positional arguments:
  video_path    Path to the video file.

options:
  -h, --help     show this help message and exit
  --start_time START_TIME
        Start time in seconds to begin extraction (default: 0)
  --end_time END_TIME 
        End time in seconds to stop extraction (default: full video)
  --extract_fps EXTRACT_FPS
        Frames per second to extract as an integer (default: 3). 
        Should typically be in [60,30,15,10,6,5,2,1] for exact match for
        typical 30/60 fps video.
  --rotate_frames {cw90,ccw90,180}
        Rotate extracted extracted_frames: cw90, ccw90, or 180 degrees (default: None)
```
