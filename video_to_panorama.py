import cv2
import numpy as np
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video to create a panoramic image."
    )

    parser.add_argument("video_path", type=str, help="Path to the video file.")

    parser.add_argument(
        "--start_time",
        type=float,
        default=0,
        help="Start time in seconds to begin extraction (default: 0)",
    )

    parser.add_argument(
        "--end_time",
        type=float,
        default=None,
        help="End time in seconds to stop extraction (default: full video)",
    )

    parser.add_argument(
        "--extract_fps",
        type=int,
        default=3,
        help="Frames per second to extract as an integer (default: 3).\n\tShould typically be in [60,30,15,10,6,5,2,1] for exact match for typical 30/60 fps video.",
    )

    parser.add_argument(
        "--rotate_frames",
        choices=["cw90", "ccw90", "180"],
        default=None,
        help="Rotate extracted extracted_frames: cw90, ccw90, or 180 degrees (default: None)",
    )

    return parser.parse_args()


# ==================================
#
FRAME_COUNT_WARNING_THRESHOLD = 30


# ==================================
#
def rotate_frames_in_place(frames: list[np.ndarray], cvd_rotation_enum: int):
    for idx, frame in enumerate(frames):
        frames[idx] = cv2.rotate(frame, cvd_rotation_enum)


# ==================================
#
def save_image_frames(
    output_dir: [Path, str],
    image_frame_list: list[np.ndarray],
    image_frame_id_list: list[int] = None,
) -> None:
    # make a place to put them
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_data_iterator = (
        zip(image_frame_id_list, image_frame_list)
        if image_frame_id_list is not None
        else enumerate(image_frame_list)
    )
    for idx, image_frame in frame_data_iterator:
        # five leading zeros covers about 9 hours without issue
        cv2.imwrite(output_dir / f"frame_{idx:05d}.png", image_frame)


# ==================================
#
def extract_frames_from_video(
    video_path: str,
    *,
    start_time: float = 0,
    end_time: float = None,
    extract_fps: int = 3,
    cv2_rotation_enum: int = None,
):
    """Extract extracted_frames between start_time and end_time in seconds."""
    assert Path(video_path).is_file()
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    video_fps_speed = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps_speed
    frame_step = int(video_fps_speed / extract_fps)
    if frame_step < 1:
        frame_step = 1

    estimated_fame_extraction = total_frames // frame_step
    print(
        f"Extracting {estimated_fame_extraction} frames between {start_time} and {end_time}."
    )

    if end_time is None or end_time > duration:
        end_time = duration

    start_frame = int(start_time * video_fps_speed)
    end_frame = int(end_time * video_fps_speed)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted_frames = []
    extracted_frame_ids = []
    # we extract them sequentially because MPG has keyframes and then interframe diffs,
    # so we have to walk the whole thing. Some documentation for reference:
    # https://en.wikipedia.org/wiki/Key_frame
    current_frame = start_frame
    while current_frame < end_frame:
        success, extracted_frame = video_capture.read()
        if not success:
            break
        if (current_frame - start_frame) % frame_step == 0:
            extracted_frames.append(extracted_frame)
            extracted_frame_ids.append(current_frame)
        current_frame += 1

    video_capture.release()

    if cv2_rotation_enum is not None:
        rotate_frames_in_place(extracted_frames, cvd_rotation_enum=cv2_rotation_enum)

    return extracted_frames, extracted_frame_ids, video_fps_speed


# ==================================
#
def stitch_and_save_panorama(frames: list[np.ndarray], image_output_path: str):

    stitcher = (
        cv2.Stitcher_create()
        if hasattr(cv2, "Stitcher_create")
        else cv2.createStitcher()
    )
    status, stitched_panorama = stitcher.stitch(frames)

    if status != cv2.Stitcher_OK:
        print(f"Stitching failed with status code: {status}")
        return

    cv2.imwrite(Path(image_output_path) / "panorama.jpg", stitched_panorama)

    # Convert stitched image to BGRA (add alpha channel)
    stitched_bgra = cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2BGRA)

    # Create a mask where pixels are black
    black_mask = cv2.inRange(stitched_panorama, (0, 0, 0), (0, 0, 0))

    # Set alpha to 0 (transparent) where mask is black
    stitched_bgra[black_mask == 255, 3] = 0

    # Save as PNG to preserve transparency
    cv2.imwrite(Path(image_output_path) / "panorama_transparent.png", stitched_bgra)


# PXL_20250614_193237148 = {
#     "video_file_path": r"C:\Users\ryzhara\Desktop\Democratic Party Things\Gallatin No Kings Rally Images\videos\PXL_20250614_193237148.mp4",
#     "start_time": 9.0,
#     "end_time": 22.0,
#     "extract_fps": 3,
#     "cv2_rotation_enum": cv2.ROTATE_90_CLOCKWISE,
# }


if __name__ == "__main__":
    # python extract_frames.py myvideo.mp4 --start_time 10 --end_time 60 --fps 5 --rotate_frames cw90
    args = parse_args()
    print(f"Video path: {args.video_path}")
    print(f"Start time: {args.start_time}")
    print(f"End time: {args.end_time if args.end_time is not None else 'end of video'}")
    print(f"Extract FPS: {args.extract_fps}")
    print(
        f"Rotate extracted_frames: {args.rotate_frames if args.rotate_frames else 'no rotation'}"
    )

    video_path = Path(args.video_path).resolve()  # get absolute path
    assert video_path.is_file()
    # this will also be the output directory
    video_path_dir = video_path.parent
    video_path_filename = video_path.stem

    assert args.extract_fps >= 1

    cv2_rotation_enum = None
    match args.rotate_frames:
        case None:
            pass
        case "cw90":
            cv2_rotation_enum = cv2.ROTATE_90_CLOCKWISE
        case "ccw90":
            cv2_rotation_enum = cv2.ROTATE_90_COUNTERCLOCKWISE
        case "180":
            cv2_rotation_enum = cv2.ROTATE_180
        case _:
            raise Exception(f"Unknown rotation mode: {args.rotate_frames}")

    output_path = (
        video_path_dir
        / f"{video_path_filename}_{args.start_time}_{args.end_time}_{args.extract_fps}"
    )
    output_frames_path = output_path / "frames"
    # ensure all the directories are created
    output_frames_path.mkdir(parents=True, exist_ok=True)

    extracted_frames, extracted_frame_ids, video_fps_speed = extract_frames_from_video(
        video_path,
        start_time=args.start_time,
        end_time=args.end_time,
        extract_fps=args.extract_fps,
        cv2_rotation_enum=cv2_rotation_enum,
    )
    # save the extracted_frames
    save_image_frames(
        output_frames_path,
        image_frame_list=extracted_frames,
        image_frame_id_list=extracted_frame_ids,
    )

    frame_count = len(extracted_frames)
    if frame_count < 2:
        raise "Not enough extracted_frames for stitching."

    print(f"Extracted {frame_count} frames from video @ {video_fps_speed} fps.")
    if frame_count > FRAME_COUNT_WARNING_THRESHOLD:
        print(
            f"warning: Stitching more than {FRAME_COUNT_WARNING_THRESHOLD} frames can take a long time."
        )
    print(f"Stitching images...")
    stitch_and_save_panorama(extracted_frames, image_output_path=output_path)
    print(f"Panorama images saved to '{output_path}'")
