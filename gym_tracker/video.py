"""Stream video decode → inference → H.264 encode with bounded frame memory."""
import math


def analyze_video(input_path, output_path, tracker, on_progress=None):
    cap = None
    writer = None
    try:
        import cv2
        import imageio.v2 as imageio
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("Unable to open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate; duration cannot be measured")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                    macro_block_size=2, ffmpeg_log_level="error")
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            processed = tracker.process(frame, count / fps)
            writer.append_data(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            if on_progress and count % 10 == 0:
                on_progress(count, total, processed)
        if count == 0:
            raise ValueError("Video contains no decodable frames")
    finally:
        if cap is not None:
            cap.release()
        try:
            if writer is not None:
                writer.close()
        finally:
            tracker.close()
    return tracker.snapshot()
