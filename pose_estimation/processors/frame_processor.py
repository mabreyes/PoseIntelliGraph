"""
Frame processor for pose estimation
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pose_estimation.models.openpose import OpenPoseModel


class FrameProcessor:
    """
    Process frames for pose estimation

    Responsible for processing individual frames to detect and visualize poses
    """

    def __init__(self, model: OpenPoseModel):
        """
        Initialize the frame processor

        Args:
            model: OpenPose model for pose detection
        """
        self.model = model

    def process(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
        """
        Process a single frame to detect poses

        Args:
            frame: Input frame

        Returns:
            Tuple containing:
                - Processed frame with visualized pose
                - List of detected keypoints
        """
        # Prepare input blob
        input_blob = self.model.prepare_input(frame)

        # Set input to the network
        if self.model.net is None:
            raise ValueError("Model network not initialized")

        self.model.net.setInput(input_blob)

        # Forward pass through the network
        output = self.model.net.forward()

        # Get keypoints from the output
        points = self.model.process_output(output, frame.shape)

        # Draw the keypoints and connections
        result_frame = self.visualize_pose(frame.copy(), points)

        return result_frame, points

    def visualize_pose(
        self, frame: np.ndarray, points: List[Optional[Tuple[int, int]]]
    ) -> np.ndarray:
        """
        Draw the detected pose on the frame

        Args:
            frame: Original frame
            points: List of detected keypoints

        Returns:
            Frame with visualized pose
        """
        # Colors for visualization - BGR format
        colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]

        # Draw the connections between keypoints
        for i, pair in enumerate(self.model.pose_pairs):
            part_from = pair[0]
            part_to = pair[1]

            id_from = self.model.body_parts[part_from]
            id_to = self.model.body_parts[part_to]

            if points[id_from] and points[id_to]:
                color_idx = i % len(colors)
                cv2.line(frame, points[id_from], points[id_to], colors[color_idx], 3)
                cv2.ellipse(
                    frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED
                )
                cv2.ellipse(
                    frame, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED
                )

        return frame
