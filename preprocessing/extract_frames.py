import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video file.
    :param video_path: Path to the video file.
    :param output_dir: Directory to save frames.
    :param frame_rate: Extract every 'n' frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Extracting {video_name}") as pbar:
        while success:
            success, image = cap.read()
            if success and count % frame_rate == 0:
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{count:04d}.jpg")
                cv2.imwrite(frame_path, image)
            count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Finished extracting {count} frames to {output_dir}")

if __name__ == "__main__":
    # Standard FaceForensics wrapper logic can be added here
    # For now, this serves as our primary extraction engine
    pass
