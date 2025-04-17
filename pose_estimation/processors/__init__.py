"""
Processors for pose estimation on frames and videos
"""
from pose_estimation.processors.directory_processor import DirectoryProcessor
from pose_estimation.processors.frame_processor import FrameProcessor
from pose_estimation.processors.video_processor import VideoProcessor

__all__ = ["FrameProcessor", "VideoProcessor", "DirectoryProcessor"]
