"""
Frame processor for pose estimation
"""
from typing import Dict, List, Optional, Tuple, cast

import cv2
import numpy as np

from pose_estimation.models.openpose import OpenPoseModel


class FrameProcessor:
    """
    Frame processor for pose estimation

    Handles processing and visualizing frames
    """

    def __init__(self, model: OpenPoseModel):
        """
        Initialize the frame processor with OpenPose model

        Args:
            model: OpenPoseModel instance
        """
        self.model = model
        self.colors = self._get_colors()

    def _get_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get limb colors for visualization"""
        colors = {
            # Torso
            "Neck-RShoulder": (0, 255, 0),  # Green
            "Neck-LShoulder": (0, 255, 0),  # Green
            "Neck-RHip": (0, 255, 0),  # Green
            "Neck-LHip": (0, 255, 0),  # Green
            "RHip-LHip": (0, 255, 0),  # Green
            # Right arm
            "RShoulder-RElbow": (255, 0, 0),  # Blue
            "RElbow-RWrist": (255, 0, 0),  # Blue
            # Left arm
            "LShoulder-LElbow": (0, 0, 255),  # Red
            "LElbow-LWrist": (0, 0, 255),  # Red
            # Right leg
            "RHip-RKnee": (255, 255, 0),  # Yellow
            "RKnee-RAnkle": (255, 255, 0),  # Yellow
            # Left leg
            "LHip-LKnee": (255, 0, 255),  # Magenta
            "LKnee-LAnkle": (255, 0, 255),  # Magenta
            # Face
            "Neck-Nose": (0, 255, 255),  # Cyan
            "Nose-REye": (0, 255, 255),  # Cyan
            "REye-REar": (0, 255, 255),  # Cyan
            "Nose-LEye": (0, 255, 255),  # Cyan
            "LEye-LEar": (0, 255, 255),  # Cyan
        }
        return colors

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame with the OpenPose model

        Args:
            frame: Input frame

        Returns:
            Processed frame with pose overlay
        """
        processed_frame = frame.copy()

        # Prepare input blob for the model
        input_blob = self.model.prepare_input(frame)

        # Check if the neural network is initialized
        if self.model.net is None:
            raise ValueError("OpenPose model network is not initialized")

        # Use cast to help mypy understand that net is not None
        net = cast(cv2.dnn.Net, self.model.net)

        # Run model inference
        net.setInput(input_blob)
        output = net.forward()

        # Process output to get keypoints
        keypoints = self.model.process_output(output, frame.shape[:2])

        # Draw the keypoints on the frame
        processed_frame = self.draw_poses(processed_frame, keypoints)

        return processed_frame

    def draw_poses(
        self, frame: np.ndarray, all_keypoints: List[List[Optional[Tuple[int, int]]]]
    ) -> np.ndarray:
        """
        Draw detected poses on the frame

        Args:
            frame: Original frame
            all_keypoints: List of keypoints for all detected people

        Returns:
            Frame with pose overlay
        """
        result_frame = frame.copy()

        # Draw each person
        for person_keypoints in all_keypoints:
            # Draw connections first (so they're behind points)
            self._draw_connections(result_frame, person_keypoints)

            # Then draw keypoints on top
            self._draw_keypoints(result_frame, person_keypoints)

        return result_frame

    def _draw_keypoints(
        self, frame: np.ndarray, keypoints: List[Optional[Tuple[int, int]]]
    ) -> None:
        """
        Draw keypoints on the frame

        Args:
            frame: Frame to draw on
            keypoints: List of keypoints
        """
        for _i, keypoint in enumerate(keypoints):
            if keypoint is not None:
                cv2.circle(
                    frame, keypoint, 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED
                )

    def _draw_connections(
        self, frame: np.ndarray, keypoints: List[Optional[Tuple[int, int]]]
    ) -> None:
        """
        Draw connections between keypoints

        Args:
            frame: Frame to draw on
            keypoints: List of keypoints
        """
        for pair in self.model.pose_pairs:
            part_a, part_b = pair
            part_a_idx = self.model.body_parts[part_a]
            part_b_idx = self.model.body_parts[part_b]

            if (
                part_a_idx < len(keypoints)
                and part_b_idx < len(keypoints)
                and keypoints[part_a_idx] is not None
                and keypoints[part_b_idx] is not None
            ):
                # Get color for this limb
                limb_key = f"{part_a}-{part_b}"
                color = self.colors.get(
                    limb_key, (255, 255, 255)
                )  # Default white if not found

                # Draw the connection line
                cv2.line(
                    frame,
                    keypoints[part_a_idx],
                    keypoints[part_b_idx],
                    color,
                    thickness=2,
                )
