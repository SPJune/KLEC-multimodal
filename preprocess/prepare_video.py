#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video preprocessing for V2SFlow-style lip reading.
Based on auto_avsr preprocessing pipeline.

Steps:
1. Convert frame rate: 30Hz -> 25Hz
2. Face detection using MediaPipe
3. Affine transformation to align face to mean face
4. Crop mouth region (96x96)
"""

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import mediapipe as mp


# Mean face landmarks path (from auto_avsr)
MEAN_FACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "auto_avsr", "preparation", "detectors", "mediapipe", "20words_mean_face.npy"
)


class LandmarksDetector:
    """MediaPipe-based face landmarks detector (from auto_avsr)."""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0
        )
        self.full_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=1
        )

    def __call__(self, video_frames):
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
        return landmarks

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].y * ih)],
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks


def linear_interpolate(landmarks, start_idx, stop_idx):
    """Interpolate missing landmarks between two valid frames."""
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def cut_patch(img, landmarks, height, width, threshold=5):
    """Cut a patch around the mouth region."""
    center_x, center_y = np.mean(landmarks, axis=0)
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise OverflowError("too much bias in height")
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise OverflowError("too much bias in width")
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    return cutted_img


class VideoProcess:
    """Process video frames: alignment and mouth cropping (from auto_avsr)."""
    
    def __init__(
        self,
        mean_face_path=MEAN_FACE_PATH,
        crop_width=96,
        crop_height=96,
        start_idx=3,
        stop_idx=4,
        window_margin=12,
        convert_gray=True,
    ):
        self.reference = np.load(mean_face_path)
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def __call__(self, video, landmarks):
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        if not preprocessed_landmarks:
            return None
        sequence = self.crop_patch(video, preprocessed_landmarks)
        return sequence

    def interpolate_landmarks(self, landmarks):
        """Interpolate frames that are not detected."""
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
        if not valid_frames_idx:
            return None
        
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )
        
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )
        
        if not all(lm is not None for lm in landmarks):
            return None
        return landmarks

    def crop_patch(self, video, landmarks):
        """Crop mouth region from each frame."""
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(
                self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx
            )
            smoothed_landmarks = np.mean(
                [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray
            )
            try:
                patch = cut_patch(
                    transformed_frame,
                    transformed_landmarks[self.start_idx : self.stop_idx],
                    self.crop_height // 2,
                    self.crop_width // 2,
                )
                sequence.append(patch)
            except OverflowError:
                return None
        return np.array(sequence)

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(0, 1, 2, 3),
    ):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        stable_reference = self.get_stable_reference(reference, reference_size, target_size)
        transform = self.estimate_affine_transform(landmarks, stable_points, stable_reference)
        
        if transform is None:
            return frame, landmarks
        
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame, landmarks, transform, target_size
        )
        return transformed_frame, transformed_landmarks

    def get_stable_reference(self, reference, reference_size, target_size):
        stable_reference = np.vstack([
            np.mean(reference[36:42], axis=0),  # right eye
            np.mean(reference[42:48], axis=0),  # left eye
            np.mean(reference[31:36], axis=0),  # nose tip
            np.mean(reference[48:68], axis=0),  # mouth center
        ])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        try:
            transform, _ = cv2.estimateAffinePartial2D(
                np.vstack([landmarks[x] for x in stable_points]),
                stable_reference,
                method=cv2.LMEDS,
            )
            return transform
        except Exception:
            return None

    def apply_affine_transform(self, frame, landmarks, transform, target_size):
        transformed_frame = cv2.warpAffine(
            frame, transform, dsize=(target_size[0], target_size[1]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        transformed_landmarks = (
            np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
        )
        return transformed_frame, transformed_landmarks


def convert_frame_rate(input_path: str, output_path: str, target_fps: int = 25):
    """Convert video frame rate from 30Hz to 25Hz using ffmpeg."""
    import subprocess
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-r', str(target_fps),
            '-vcodec', 'libx264', '-crf', '18', '-preset', 'fast',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error converting frame rate: {e}")
        return False


def load_video(video_path: str) -> list:
    """Load video frames as a list of numpy arrays (RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames: np.ndarray, output_path: str, fps: int = 25):
    """Save processed frames as a video file."""
    if frames is None or len(frames) == 0:
        return False
    
    h, w = frames[0].shape[:2]
    is_gray = len(frames[0].shape) == 2
    
    # For grayscale, convert to BGR for proper mp4 encoding
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)
    
    for frame in frames:
        if is_gray:
            # Convert grayscale to BGR for VideoWriter
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    
    out.release()
    
    # Verify file was written
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True
    return False


def preprocess_single_video(
    input_path: str,
    output_path: str,
    detector: LandmarksDetector,
    processor: VideoProcess,
    target_fps: int = 25,
    source_fps: int = 30,
):
    """Preprocess a single video file."""
    # Create temp path for fps-converted video
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    
    # Step 1: Convert frame rate if needed
    if source_fps != target_fps:
        if not convert_frame_rate(input_path, temp_path, target_fps):
            print(f"Failed to convert frame rate: {input_path}")
            return False
        video_to_process = temp_path
    else:
        video_to_process = input_path
    
    # Step 2: Load video frames
    frames = load_video(video_to_process)
    if len(frames) == 0:
        print(f"Failed to load video: {video_to_process}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    
    # Step 3: Detect landmarks
    landmarks = detector(frames)
    if all(lm is None for lm in landmarks):
        print(f"No face detected in video: {input_path}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    
    # Step 4: Process video (align and crop mouth)
    processed_frames = processor(frames, landmarks)
    if processed_frames is None or len(processed_frames) == 0:
        print(f"Failed to process video: {input_path}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    
    # Step 5: Save processed video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = save_video(processed_frames, output_path, fps=target_fps)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return success


def get_output_path(base_path: str, data_type: str, sess: str, input_filename: str) -> str:
    """Generate output path for preprocessed video."""
    output_dir = os.path.join(base_path, data_type, sess, 'data/video_preprocessed')
    return os.path.join(output_dir, input_filename)


def preprocess_dataset(
    base_path: str,
    data_types: list = None,
    sessions: list = None,
    source_fps: int = 30,
    target_fps: int = 25,
    convert_gray: bool = True,
):
    """Preprocess all videos in the dataset."""
    if data_types is None:
        data_types = ['voiced', 'silent']
    
    detector = LandmarksDetector()
    processor = VideoProcess(convert_gray=convert_gray)
    
    # Find all video files
    video_files = []
    for data_type in data_types:
        data_type_path = os.path.join(base_path, data_type)
        if not os.path.exists(data_type_path):
            continue
        
        for sess in os.listdir(data_type_path):
            if sessions is not None and sess not in sessions:
                continue
            video_dir = os.path.join(data_type_path, sess, 'data/video')
            if not os.path.exists(video_dir):
                continue
            
            for video_file in glob(os.path.join(video_dir, '*.mp4')):
                output_path = get_output_path(
                    base_path, data_type, sess, os.path.basename(video_file)
                )
                if not os.path.exists(output_path):
                    video_files.append((video_file, output_path))
    
    print(f"Found {len(video_files)} videos to preprocess")
    
    # Process videos
    success_count = 0
    for input_path, output_path in tqdm(video_files, desc="Preprocessing videos"):
        if preprocess_single_video(
            input_path, output_path, detector, processor,
            target_fps=target_fps, source_fps=source_fps
        ):
            success_count += 1
    
    print(f"Successfully preprocessed {success_count}/{len(video_files)} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video preprocessing for V2SFlow")
    parser.add_argument("--base-path", type=str, default=None,
                        help="Base path to the dataset (e.g., /path/to/silent_speech_dataset)")
    parser.add_argument("--data-types", type=str, nargs="+", default=["voiced", "silent"],
                        help="Data types to process")
    parser.add_argument("--sessions", type=str, nargs="+", default=None,
                        help="Specific sessions to process (default: all)")
    parser.add_argument("--source-fps", type=int, default=30,
                        help="Source video frame rate (default: 30)")
    parser.add_argument("--target-fps", type=int, default=25,
                        help="Target video frame rate (default: 25)")
    parser.add_argument("--color", action="store_true",
                        help="Keep color instead of converting to grayscale")
    
    # Single video mode
    parser.add_argument("--input", type=str, default=None,
                        help="Single input video path")
    parser.add_argument("--output", type=str, default=None,
                        help="Single output video path")
    
    args = parser.parse_args()
    
    if args.input and args.output:
        # Single video mode
        detector = LandmarksDetector()
        processor = VideoProcess(convert_gray=not args.color)
        success = preprocess_single_video(
            args.input, args.output, detector, processor,
            target_fps=args.target_fps, source_fps=args.source_fps
        )
        if success:
            print(f"Successfully preprocessed: {args.output}")
        else:
            print(f"Failed to preprocess: {args.input}")
            sys.exit(1)
    elif args.base_path:
        # Dataset mode
        preprocess_dataset(
            base_path=args.base_path,
            data_types=args.data_types,
            sessions=args.sessions,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            convert_gray=not args.color,
        )
    else:
        parser.print_help()
        print("\nError: Either provide --input and --output for single video mode,")
        print("       or --base-path for dataset mode.")
        sys.exit(1)
