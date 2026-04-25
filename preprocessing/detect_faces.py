import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image

class FaceDetector:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # margin=20 adds some skin/context around the face, critical for deepfake detection
        self.mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=self.device)
        print(f"Using device: {self.device}")

    def crop_face(self, image_path, save_path):
        """Detects and crops a single face from an image."""
        try:
            img = Image.open(image_path).convert('RGB')
            # Detect and save to save_path
            self.mtcnn(img, save_path=save_path)
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False

    def process_directory(self, input_dir, output_dir):
        """Processes all frames in a directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Found {len(images)} frames in {input_dir}")

        for img_name in tqdm(images, desc="Cropping Faces"):
            input_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, img_name)
            self.crop_face(input_path, output_path)

if __name__ == "__main__":
    # Example usage
    detector = FaceDetector()
    # Replace these paths as needed during execution
    # detector.process_directory('data/raw_frames', 'data/face_crops')
