"""
Command-line interface for the pose estimation application
"""
import argparse
from pathlib import Path

from pose_estimation.models.openpose import OpenPoseModel
from pose_estimation.processors.video_processor import VideoProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Video Pose Estimation using OpenPose")

    parser.add_argument(
        "input", help="Path to input video file or 'camera' to use webcam"
    )

    parser.add_argument("-o", "--output", help="Path to output video file")

    parser.add_argument("-m", "--model-path", help="Path to OpenPose model directory")

    parser.add_argument(
        "-nd",
        "--no-display",
        action="store_true",
        help="Don't display output while processing",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI application"""
    args = parse_args()

    # Initialize model
    model_path = Path(args.model_path) if args.model_path else None
    model = OpenPoseModel(model_path=model_path)

    # Initialize video processor
    processor = VideoProcessor(model)

    # Process the video
    processor.process_video(
        args.input, output_path=args.output, display=not args.no_display
    )


if __name__ == "__main__":
    main()
