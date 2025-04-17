"""
OpenPose model loader and configuration
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class OpenPoseModel:
    """
    OpenPose model loader and configuration

    Responsible for loading and configuring the OpenPose model
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the OpenPose model

        Args:
            model_path: Path to the model directory. If None, uses the default location.
        """
        self.model_path = Path(model_path) if model_path else Path("models")
        self.protofile_path = self.model_path / "pose/coco/pose_deploy_linevec.prototxt"
        self.weights_path = self.model_path / "pose/coco/pose_iter_440000.caffemodel"
        self.net: Optional[cv2.dnn.Net] = None
        self.input_width = 368
        self.input_height = 368
        self.threshold = 0.1
        # Higher for better multiple person detection
        self.nms_threshold = 0.15

        # Check if model files exist
        self._validate_model_files()

        # Load the model
        self._load_model()

        # Define the body parts and connections
        self.body_parts = self._get_body_parts()
        self.pose_pairs = self._get_pose_pairs()

        # Total number of parts in COCO model
        self.num_points = 18

        # PAF (part affinity fields) indices for each pose pair
        self.map_idx = [
            [31, 32],
            [39, 40],
            [33, 34],
            [35, 36],
            [41, 42],
            [43, 44],
            [19, 20],
            [21, 22],
            [23, 24],
            [25, 26],
            [27, 28],
            [29, 30],
            [47, 48],
            [49, 50],
            [53, 54],
            [51, 52],
            [55, 56],
        ]

    def _validate_model_files(self) -> None:
        """Validate that the model files exist"""
        if not self.protofile_path.exists() or not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model files not found. Please ensure the files exist at:\n"
                f"  - {self.protofile_path}\n"
                f"  - {self.weights_path}"
            )

    def _load_model(self) -> None:
        """Load the OpenPose model"""
        try:
            self.net = cv2.dnn.readNetFromCaffe(
                str(self.protofile_path), str(self.weights_path)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenPose model: {e}") from e

    def _get_body_parts(self) -> Dict[str, int]:
        """Define the body parts for COCO dataset"""
        return {
            "Nose": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "RHip": 8,
            "RKnee": 9,
            "RAnkle": 10,
            "LHip": 11,
            "LKnee": 12,
            "LAnkle": 13,
            "REye": 14,
            "LEye": 15,
            "REar": 16,
            "LEar": 17,
            "Background": 18,
        }

    def _get_pose_pairs(self) -> List[List[str]]:
        """Define the pose pairs that connect the body parts"""
        return [
            ["Neck", "RShoulder"],
            ["Neck", "LShoulder"],
            ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"],
            ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"],
            ["Neck", "RHip"],
            ["RHip", "RKnee"],
            ["RKnee", "RAnkle"],
            ["Neck", "LHip"],
            ["LHip", "LKnee"],
            ["LKnee", "LAnkle"],
            ["Neck", "Nose"],
            ["Nose", "REye"],
            ["REye", "REar"],
            ["Nose", "LEye"],
            ["LEye", "LEar"],
        ]

    def prepare_input(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a frame for input to the network

        Args:
            frame: Input frame

        Returns:
            Prepared blob for network input
        """
        return cv2.dnn.blobFromImage(
            frame,
            1.0 / 255,
            (self.input_width, self.input_height),
            (0, 0, 0),
            swapRB=False,
            crop=False,
        )

    def process_output(
        self, output: np.ndarray, frame_shape: Tuple[int, int]
    ) -> List[List[Optional[Tuple[int, int]]]]:
        """
        Process network output to get keypoints for multiple people

        Args:
            output: Network output
            frame_shape: Original frame shape (height, width)

        Returns:
            List of people, each containing a list of keypoints (x, y)
            or None if not detected
        """
        frame_height, frame_width = frame_shape[:2]
        detected_keypoints: List[List[Tuple[int, int, int, float]]] = []
        keypoint_id = 0

        # Initialize empty lists for each body part for all people
        for _ in range(self.num_points):
            detected_keypoints.append([])

        # For each body part, find keypoints for multiple people
        for part_id in range(self.num_points):
            # Get probability map for current part
            prob_map = output[0, part_id, :, :]
            prob_map = cv2.resize(prob_map, (frame_width, frame_height))

            # Find all keypoints above threshold
            keypoints = self._get_keypoints(prob_map, self.threshold)

            # Add a keypoint ID to each keypoint
            keypoints_with_id = []
            for i, keypoint in enumerate(keypoints):
                keypoints_with_id.append(
                    (keypoint[0], keypoint[1], keypoint_id + i, keypoint[2])
                )

            # Update keypoint ID for next part
            keypoint_id += len(keypoints)

            # Add detected keypoints for this part
            detected_keypoints[part_id] = keypoints_with_id

        # Use PAF (Part Affinity Fields) to group keypoints into people
        valid_pairs, invalid_pairs = self._get_valid_pairs(
            output, detected_keypoints, frame_shape
        )

        # Assemble all the parts for each person
        person_keypoints = self._get_personwise_keypoints(
            valid_pairs, invalid_pairs, detected_keypoints
        )

        # Convert to list of keypoints per person
        # Each person has a list of points (possibly None for missing parts)
        keypoints_list = []
        for i in range(len(detected_keypoints)):
            for kpt in detected_keypoints[i]:
                keypoints_list.append([kpt[0], kpt[1], kpt[3]])  # x, y, score

        keypoints_list = np.array(keypoints_list)

        # Convert to list of keypoints per person
        result = []
        for person in person_keypoints:
            # Skip persons with very low confidence
            if person[-1] < 1:
                continue

            # Extract keypoints for this person
            person_points: List[Optional[Tuple[int, int]]] = []
            for i in range(self.num_points):
                keypoint_idx = int(person[i])
                if keypoint_idx == -1:  # Missing keypoint
                    person_points.append(None)
                else:
                    # Get x, y from the keypoint id
                    found = False
                    for kpt in detected_keypoints[i]:
                        if kpt[2] == keypoint_idx:
                            person_points.append((int(kpt[0]), int(kpt[1])))
                            found = True
                            break
                    if not found:
                        person_points.append(None)

            result.append(person_points)

        return result

    def _get_keypoints(
        self, prob_map: np.ndarray, threshold: float = 0.1
    ) -> List[Tuple[int, int, float]]:
        """
        Extract keypoints from probability map

        Args:
            prob_map: Probability map for a body part
            threshold: Confidence threshold

        Returns:
            List of keypoints (x, y, probability)
        """
        map_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)
        map_mask = np.uint8(map_smooth > threshold)
        keypoints: List[Tuple[int, int, float]] = []

        # Find connected components (blobs)
        contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # For each blob, find the maxima
        for contour in contours:
            mask = np.zeros(map_mask.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, contour, (1,))
            masked_prob_map = map_smooth * mask
            _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
            keypoints.append(
                (
                    int(max_loc[0]),
                    int(max_loc[1]),
                    float(prob_map[max_loc[1], max_loc[0]]),
                )
            )

        return keypoints

    def _get_valid_pairs(
        self,
        output: np.ndarray,
        detected_keypoints: List[List[Tuple[int, int, int, float]]],
        frame_shape: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Find valid connections between keypoints

        Args:
            output: Network output
            detected_keypoints: List of keypoints for each part
            frame_shape: Original frame shape

        Returns:
            Tuple of valid pairs and invalid pairs
        """
        frame_height, frame_width = frame_shape[:2]

        valid_pairs: List[np.ndarray] = []
        invalid_pairs: List[int] = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7

        # For each pose pair, find valid connections
        for k, (body_part_a, body_part_b) in enumerate(self.pose_pairs):
            part_idx_a = self.body_parts[body_part_a]
            part_idx_b = self.body_parts[body_part_b]

            # Get PAF for this limb
            paf_a = output[0, self.map_idx[k][0], :, :]
            paf_b = output[0, self.map_idx[k][1], :, :]
            paf_a = cv2.resize(paf_a, (frame_width, frame_height))
            paf_b = cv2.resize(paf_b, (frame_width, frame_height))

            # Get keypoints for parts A and B
            candidate_a = detected_keypoints[part_idx_a]
            candidate_b = detected_keypoints[part_idx_b]

            n_a = len(candidate_a)
            n_b = len(candidate_b)

            # If keypoints for the joint pair are detected
            if n_a != 0 and n_b != 0:
                valid_pair = np.zeros((0, 3))

                # For each joint in candidateA
                for i, kpt_a in enumerate(candidate_a):
                    max_j = -1
                    max_score = -1
                    found = 0

                    # For each joint in candidateB
                    for j, kpt_b in enumerate(candidate_b):
                        # Calculate distance vector between A and B
                        d_x = kpt_b[0] - kpt_a[0]
                        d_y = kpt_b[1] - kpt_a[1]
                        norm = np.sqrt(d_x * d_x + d_y * d_y)

                        if norm == 0:
                            continue

                        # Normalize
                        d_x = d_x / norm
                        d_y = d_y / norm

                        # Get points along line for PAF evaluation
                        interp_coord = []
                        for r in range(n_interp_samples):
                            x = int(
                                kpt_a[0]
                                + r * (kpt_b[0] - kpt_a[0]) / (n_interp_samples - 1)
                            )
                            y = int(
                                kpt_a[1]
                                + r * (kpt_b[1] - kpt_a[1]) / (n_interp_samples - 1)
                            )
                            interp_coord.append((x, y))

                        # Evaluate PAF along the line
                        paf_interp = []
                        for interp_point in interp_coord:
                            x, y = interp_point
                            if x < 0 or y < 0 or x >= frame_width or y >= frame_height:
                                continue
                            paf_interp.append([paf_a[y, x], paf_b[y, x]])

                        # Find avg PAF score along line
                        paf_scores = np.dot(paf_interp, np.array([d_x, d_y]))
                        avg_paf_score = (
                            sum(paf_scores) / len(paf_scores)
                            if len(paf_scores) > 0
                            else 0
                        )

                        # Check if connection is valid
                        if (
                            len(paf_scores) > 0
                            and (
                                len(np.where(paf_scores > paf_score_th)[0])
                                / len(paf_scores)
                            )
                            > conf_th
                        ):
                            if avg_paf_score > max_score:
                                max_j = j
                                max_score = avg_paf_score
                                found = 1

                    # Add connection to valid pairs
                    if found:
                        valid_pair = np.append(
                            valid_pair,
                            [[candidate_a[i][2], candidate_b[max_j][2], max_score]],
                            axis=0,
                        )

                # Append the detected connections to the list
                valid_pairs.append(valid_pair)
            else:
                # No connections for this pair
                valid_pairs.append(np.array([]))
                invalid_pairs.append(k)

        return valid_pairs, invalid_pairs

    def _get_personwise_keypoints(
        self,
        valid_pairs: List[np.ndarray],
        invalid_pairs: List[int],
        detected_keypoints: List[List[Tuple[int, int, int, float]]],
    ) -> NDArray:
        """
        Assemble detected parts into people

        Args:
            valid_pairs: Valid connections between parts
            invalid_pairs: Invalid connections
            detected_keypoints: Detected keypoints for each part

        Returns:
            Array of people, each row is a person with indices to keypoints
        """
        # Keep track of keypoint IDs assigned to people
        personwise_keypoints = (
            np.ones((0, 19), dtype=np.float64) * -1
        )  # 18 keypoints + overall score

        # For each type of limb connection
        for k in range(len(self.pose_pairs)):
            if k in invalid_pairs:
                continue

            part_a_name, part_b_name = self.pose_pairs[k]
            part_a_id = self.body_parts[part_a_name]
            part_b_id = self.body_parts[part_b_name]

            # If no valid pairs for this limb
            if len(valid_pairs[k]) == 0:
                continue

            # Get the valid pairs for this limb
            for pair_id in range(len(valid_pairs[k])):
                found = False
                person_idx = -1

                # Get IDs for the points
                part_a_idx = float(valid_pairs[k][pair_id][0])
                part_b_idx = float(valid_pairs[k][pair_id][1])

                # For each person detected so far
                for i in range(len(personwise_keypoints)):
                    # If part A is already assigned to this person
                    if personwise_keypoints[i][part_a_id] == part_a_idx:
                        # Assign part B to this person
                        person_idx = i
                        found = True
                        break

                # If this part wasn't assigned to any person yet
                if not found and k < len(self.pose_pairs) - 1:
                    # Create a new person
                    row = np.ones(19, dtype=np.float64) * -1
                    row[part_a_id] = part_a_idx
                    row[part_b_id] = part_b_idx

                    # Add the keypoint scores
                    score = 0.0
                    for kpt in detected_keypoints[part_a_id]:
                        if kpt[2] == int(part_a_idx):
                            score += kpt[3]
                            break

                    for kpt in detected_keypoints[part_b_id]:
                        if kpt[2] == int(part_b_idx):
                            score += kpt[3]
                            break

                    row[-1] = score + valid_pairs[k][pair_id][2]
                    personwise_keypoints = np.vstack([personwise_keypoints, row])
                # If found, update the person
                elif found:
                    personwise_keypoints[person_idx][part_b_id] = part_b_idx
                    # Add score
                    personwise_keypoints[person_idx][-1] += valid_pairs[k][pair_id][2]
                    for kpt in detected_keypoints[part_b_id]:
                        if kpt[2] == int(part_b_idx):
                            personwise_keypoints[person_idx][-1] += kpt[3]
                            break

        return personwise_keypoints
