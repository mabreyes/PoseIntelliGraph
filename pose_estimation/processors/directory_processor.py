"""
Directory processor for batch processing videos in a directory
"""
import os
from typing import Optional

from pose_estimation.models.openpose import OpenPoseModel
from pose_estimation.processors.video_processor import VideoProcessor


class DirectoryProcessor:
    """
    Process videos in a directory for pose estimation

    Responsible for batch processing multiple video files in a directory
    """

    def __init__(self, model: OpenPoseModel):
        """
        Initialize the directory processor

        Args:
            model: OpenPose model for pose detection
        """
        self.model = model
        self.video_processor = VideoProcessor(model)

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        display: bool = False,
    ) -> None:
        """
        Process all videos in a directory

        Args:
            input_dir: Path to input directory containing videos
            output_dir: Path to output directory for processed videos
            display: Whether to display output while processing
        """
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = f"{input_dir}_openpose"

        # At this point, output_dir is guaranteed to be a string
        output_dir_str: str = output_dir
        os.makedirs(output_dir_str, exist_ok=True)

        # Get list of video files in the directory
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        video_files = []

        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)

        if not video_files:
            print(f"No video files found in {input_dir}")
            return

        print(f"Found {len(video_files)} video files to process")

        # Process each video
        for i, video_file in enumerate(video_files):
            input_path = os.path.join(input_dir, video_file)

            # Create output path with same filename but in output directory
            base_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(output_dir_str, f"{base_name}_processed.mp4")

            print(f"Processing video {i+1}/{len(video_files)}: {video_file}")

            # Process the video
            self.video_processor.process_video(
                input_path, output_path=output_path, display=display
            )

        print(f"All videos processed. Results saved to {output_dir_str}")
