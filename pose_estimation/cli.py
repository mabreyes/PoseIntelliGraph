"""
Command-line interface for the pose estimation application
"""
import argparse
import os
from pathlib import Path

from pose_estimation.models.openpose import OpenPoseModel
from pose_estimation.processors.directory_processor import DirectoryProcessor
from pose_estimation.processors.video_processor import VideoProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Video Pose Estimation using OpenPose")

    parser.add_argument(
        "input",
        help="Path to input video file, directory of videos, or 'camera' to use webcam",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path to output video file (required for single video input)",
    )

    parser.add_argument("-m", "--model-path", help="Path to OpenPose model directory")

    parser.add_argument(
        "-nd",
        "--no-display",
        action="store_true",
        help="Don't display output while processing",
    )

    parser.add_argument(
        "--model-type",
        choices=["openpose"],
        default="openpose",
        help="Type of pose estimation model to use (only OpenPose supported)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI application"""
    args = parse_args()
    input_path = args.input

    # Initialize model
    model_path = Path(args.model_path) if args.model_path else None
    model = OpenPoseModel(model_path=model_path)

    # Check if the input is a directory
    if os.path.isdir(input_path):
        # Process all videos in directory
        output_dir = args.output if args.output else None

        # Initialize directory processor
        processor = DirectoryProcessor(model)

        # Process all videos in the directory
        processor.process_directory(
            input_path,
            output_dir=output_dir,
            display=False,  # Don't display when processing a directory
        )
    else:
        # Single video processing
        if not args.output and input_path != "camera":
            print("Error: Output path is required for single video input")
            return

        # Initialize video processor
        processor = VideoProcessor(model)

        # Process the video
        processor.process_video(
            args.input, output_path=args.output, display=not args.no_display
        )


if __name__ == "__main__":
    main()
