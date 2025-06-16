# CrowdCounterTools
Simple tools to count crowd density at public gatherings

```commandline
# To extract frames from a video that cover a smooth panorama:
python extract_frames.py myvideo.mp4 --start_time 10 --end_time 20 --fps 3 --rotate_frames cw90
```
And in between, use your favorite video editor and human vision, or automation
to tag stuff in your panoramic image with 'blobs' of paint, whether from the 
video extraxtion or other source, that need to be counted with a known color.
Say, RED=(255,0,0). Then you can use the blob counter to count the markers.

```commandline
# To count ares of a known color in an image and save an image with what's been marked:
python blob_counter.py path_to_image.jpg --target_rgb red --tolerance 20 --show_work true --marker_color green
```
The blob counter tool works on whatever OpenCV can read. It has been tested on both PNG and JPG images.

## Panorama Syntax
```commandline
usage: video_to_panorama.py [-h] 
        [--start_time START_TIME] 
        [--end_time END_TIME] 
        [--extract_fps EXTRACT_FPS] 
        [--rotate_frames {cw90,ccw90,180}]
        video_path

Extract frames from a video to create a panoramic image.

positional arguments:
  video_path            Path to the video file.

options:
  -h, --help            show this help message and exit
  --start_time START_TIME
                        Start time in seconds to begin extraction (default: 0)
  --end_time END_TIME   End time in seconds to stop extraction (default: full video)
  --extract_fps EXTRACT_FPS
                        Frames per second to extract as an integer (default: 3). 
                        Should typically be in [60,30,15,10,6,5,2,1] for exact match for
                        typical 30/60 fps video.
  --rotate_frames {cw90,ccw90,180}
                        Rotate extracted extracted_frames: cw90, ccw90, or 180 degrees (default: None)
```

## Blob Counter Syntax
```commandline
usage: blob_counter.py [-h] 
        --target_rgb TARGET_RGB 
        [--tolerance TOLERANCE] 
        [--show_work SHOW_WORK] 
        [--marker_color MARKER_COLOR] 
        image_path

Count and visualize RGB blobs in an image.

positional arguments:
  image_path            Path to the input image

options:
  -h, --help            show this help message and exit
  --target_rgb TARGET_RGB
                        Target color as 'red', 'green', etc. or RGB tuple like 255,0,0
  --tolerance TOLERANCE
                        A measure of color-match leeway for blob identification (default: 30)
  --show_work SHOW_WORK
                        Whether to draw contours: 'true', 'false' (default), or 'grayscale'
  --marker_color MARKER_COLOR
                        Color for contour marker as name or RGB tuple (default: green)
```