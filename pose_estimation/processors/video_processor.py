"""
Video processor for pose estimation
"""
import os
from typing import Optional, Union

import cv2

from pose_estimation.models.openpose import OpenPoseModel
from pose_estimation.processors.frame_processor import FrameProcessor


class VideoProcessor:
    """
    Process videos for pose estimation

    Responsible for processing video files or camera feeds to detect poses
    """

    def __init__(self, model: OpenPoseModel):
        """
        Initialize the video processor

        Args:
            model: OpenPose model for pose detection
        """
        self.model = model
        self.frame_processor = FrameProcessor(model)

    def process_video(
        self,
        input_path: Union[str, int],
        output_path: Optional[str] = None,
        display: bool = True,
    ) -> None:
        """
        Process video file or camera feed and apply pose estimation

        Args:
            input_path: Path to input video file or 'camera' to use webcam
            output_path: Path to output video file
            display: Whether to display output while processing
        """
        # Ensure the output directory exists if output_path is specified
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        # Open video or camera
        cap = self._open_video_capture(input_path)
        if cap is None:
            return

        # Setup output video writer if specified
        out = self._setup_video_writer(cap, output_path)

        # Process video frames
        self._process_frames(cap, out, input_path, display)

        # Clean up
        self._cleanup(cap, out, output_path)

    def _open_video_capture(
        self, input_path: Union[str, int]
    ) -> Optional[cv2.VideoCapture]:
        """
        Open video capture for file or camera

        Args:
            input_path: Path to input video file or 'camera' to use webcam

        Returns:
            Video capture object or None if failed
        """
        if input_path == "camera" or input_path == 0:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            print(f"Error: Could not open video source {input_path}")
            return None

        return cap

    def _setup_video_writer(
        self, cap: cv2.VideoCapture, output_path: Optional[str]
    ) -> Optional[cv2.VideoWriter]:
        """
        Setup video writer for output

        Args:
            cap: Video capture object
            output_path: Path to output video file

        Returns:
            Video writer object or None if no output path
        """
        if not output_path:
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # The cv2.VideoWriter_fourcc function is dynamically defined by OpenCV
        # Use getattr to access it for type checker compatibility
        fourcc_func = cv2.VideoWriter_fourcc
        fourcc = fourcc_func(*"mp4v")

        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def _process_frames(
        self,
        cap: cv2.VideoCapture,
        out: Optional[cv2.VideoWriter],
        input_path: Union[str, int],
        display: bool,
    ) -> None:
        """
        Process frames from video capture

        Args:
            cap: Video capture object
            out: Video writer object
            input_path: Original input path
            display: Whether to display output
        """
        frame_idx = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = self.frame_processor.process_frame(frame)

            # Write to output video
            if out:
                out.write(processed_frame)

            # Display the frame
            if display:
                cv2.imshow("Pose Estimation", processed_frame)

                # Progress info for video files
                if input_path != "camera" and input_path != 0:
                    print(f"Processing: {frame_idx+1}/{frame_count} frames", end="\r")

            # Break on 'q' key press
            if display and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

            frame_idx += 1

    def _cleanup(
        self,
        cap: cv2.VideoCapture,
        out: Optional[cv2.VideoWriter],
        output_path: Optional[str],
    ) -> None:
        """
        Clean up resources

        Args:
            cap: Video capture object
            out: Video writer object
            output_path: Path to output video file
        """
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        if output_path:
            print(f"\nProcessed video saved to: {output_path}")
