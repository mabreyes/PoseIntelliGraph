"""
Definitions of anatomical connections and constants for pose graph construction.

This module contains lists of predefined connections between keypoints,
representing different parts of the human body like limbs, face, and hands.
These are used to construct the edges of the pose graph.

Keypoint indices are based on the MMPose format, specifically the COCO-WholeBody layout.
Refer to MMPose documentation for details on keypoint indexing:
https://mmpose.readthedocs.io/en/latest/dataset_zoo/coco_wholebody.html
"""

# Keypoint indices for COCO-WholeBody:
# 0-16: COCO keypoints (body)
#   0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
#   5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
#   9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
#   13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
# 17-22: Foot keypoints (not explicitly used in these connection sets)
# 23-90: Face keypoints (68 points)
#   Face contour starts at 23. Specific landmark indices depend on the exact model (e.g., 68-point standard).
#   Nose (0) is part of body keypoints and often used as a central face anchor.
# 91-111: Left hand keypoints (21 points)
#   91: left_hand_wrist (root)
#   Thumb: 92(CMC), 93(MCP), 94(IP), 95(TIP)
#   Index: 96(MCP), 97(PIP), 98(DIP), 99(TIP)
#   ... and so on for Middle, Ring, Pinky.
# 112-132: Right hand keypoints (21 points)
#   112: right_hand_wrist (root)
#   Thumb: 113(CMC), 114(MCP), 115(IP), 116(TIP)
#   ... and so on.

# --------------------------
# Main Body Connections
# --------------------------
BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head (nose to eyes, eyes to ears)
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso and arms
    (5, 11), (6, 12), (11, 12), # Torso to hips
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (0, 5), (0, 6) # Neck (nose to shoulders as proxy)
]

# --------------------------
# Face Connections and Constants
# --------------------------
FACE_KEYPOINTS_START_IDX = 23
FACE_KEYPOINTS_END_IDX = 90
FACE_CENTER_POINT_IDX = 0  # Nose is often used as a central anchor for the face

# FACE_TO_BODY_CONNECTIONS: Connects a primary face point to a primary body point.
# Example: Connect first point of face contour (23) to nose (0).
FACE_TO_BODY_CONNECTIONS = [
    (FACE_CENTER_POINT_IDX, FACE_KEYPOINTS_START_IDX)
]

# --------------------------
# Hand Connections and Constants
# --------------------------
LEFT_HAND_ROOT_IDX = 91   # Wrist point for the left hand (COCO-WholeBody)
RIGHT_HAND_ROOT_IDX = 112 # Wrist point for the right hand (COCO-WholeBody)

COCO_LEFT_WRIST_IDX = 9
COCO_RIGHT_WRIST_IDX = 10

# LEFT_HAND_TO_BODY: Connects left hand root (wrist) to the corresponding body wrist keypoint.
LEFT_HAND_TO_BODY = [
    (COCO_LEFT_WRIST_IDX, LEFT_HAND_ROOT_IDX)
]

# RIGHT_HAND_TO_BODY: Connects right hand root (wrist) to the corresponding body wrist keypoint.
RIGHT_HAND_TO_BODY = [
    (COCO_RIGHT_WRIST_IDX, RIGHT_HAND_ROOT_IDX)
]

# Left Hand Finger Connections (sequential joints for each finger)
# Each finger connects from wrist -> CMC/MCP -> PIP -> DIP -> TIP (structure varies slightly)
# Using the standard 21 keypoints per hand model.
LEFT_THUMB_FINGER = [LEFT_HAND_ROOT_IDX, 92, 93, 94, 95]
LEFT_INDEX_FINGER = [LEFT_HAND_ROOT_IDX, 96, 97, 98, 99]
LEFT_MIDDLE_FINGER = [LEFT_HAND_ROOT_IDX, 100, 101, 102, 103]
LEFT_RING_FINGER = [LEFT_HAND_ROOT_IDX, 104, 105, 106, 107]
LEFT_PINKY_FINGER = [LEFT_HAND_ROOT_IDX, 108, 109, 110, 111]
LEFT_FINGERS = [
    LEFT_THUMB_FINGER, LEFT_INDEX_FINGER, LEFT_MIDDLE_FINGER,
    LEFT_RING_FINGER, LEFT_PINKY_FINGER
]

# Right Hand Finger Connections
RIGHT_THUMB_FINGER = [RIGHT_HAND_ROOT_IDX, 113, 114, 115, 116]
RIGHT_INDEX_FINGER = [RIGHT_HAND_ROOT_IDX, 117, 118, 119, 120]
RIGHT_MIDDLE_FINGER = [RIGHT_HAND_ROOT_IDX, 121, 122, 123, 124]
RIGHT_RING_FINGER = [RIGHT_HAND_ROOT_IDX, 125, 126, 127, 128]
RIGHT_PINKY_FINGER = [RIGHT_HAND_ROOT_IDX, 129, 130, 131, 132]
RIGHT_FINGERS = [
    RIGHT_THUMB_FINGER, RIGHT_INDEX_FINGER, RIGHT_MIDDLE_FINGER,
    RIGHT_RING_FINGER, RIGHT_PINKY_FINGER
]

# Keypoints at the base of each finger (MCP joints), used for connecting across the palm.
LEFT_FINGER_BASES = [92, 96, 100, 104, 108]  # Thumb CMC, then MCPs for other fingers from original code
RIGHT_FINGER_BASES = [113, 117, 121, 125, 129]  # Thumb CMC, then MCPs for other fingers

# --------------------------
# Grouping Connections
# --------------------------
# This dictionary can be used to iterate through all standard connection sets.
ALL_CONNECTION_GROUPS = {
    "body": BODY_CONNECTIONS,
    "face_to_body": FACE_TO_BODY_CONNECTIONS,
    "left_hand_to_body": LEFT_HAND_TO_BODY,
    "right_hand_to_body": RIGHT_HAND_TO_BODY,
    # Finger connections are handled separately by iterating through LEFT_FINGERS and RIGHT_FINGERS lists.
    # Inter-finger-base connections are also handled separately.
}

# --------------------------
# Keypoint Range Constants
# --------------------------
# These define the slice indices for different parts from a flat list of keypoints.
# BODY_KEYPOINTS_COCO_STANDARD = list(range(17)) # 0-16
# FACE_KEYPOINTS_RANGE = (23, 90) # MMPose WholeBody face (inclusive)
# LEFT_HAND_KEYPOINTS_RANGE = (91, 111) # MMPose WholeBody left hand (inclusive)
# RIGHT_HAND_KEYPOINTS_RANGE = (112, 132) # MMPose WholeBody right hand (inclusive)
# Note: The ranges above are more for semantic understanding. The connection lists define actual topology.

# --------------------------
# Configuration for Graph Creation
# --------------------------
# Threshold for determining if a keypoint is "valid" based on its confidence score.
# Keypoints with confidence below this threshold will be excluded from the graph.
MIN_CONFIDENCE_THRESHOLD = 0.3  # Example value, can be tuned.

# For connecting sparse regions or ensuring graph connectivity:
# If the graph is detected as sparse or having disconnected components,
# these constants can guide how to add more edges.
# NUM_NEAREST_ANCHORS = 2 # Example: connect a sparse node to its 2 nearest anchor points.
# MAIN_BODY_ANCHOR_KEYPOINTS = [0, 5, 6, 9, 10, 11, 12] # Nose, Shoulders, Wrists, Hips
# The current refactored `create_pose_graph` uses a simpler sparsity check.
# These constants are kept for potential future enhancements.

# Note: The original `create_pose_graph` had logic for:
# 1. Connecting face keypoints sequentially (e.g., 23-24, 24-25, ...).
# 2. Connecting all face keypoints to a central point (nose, index 0).
# 3. Connecting finger joints sequentially (e.g., wrist-MCP, MCP-PIP, ... for each finger).
# 4. Connecting finger bases to each other (e.g., left_thumb_base to left_index_base, etc.).
# This logic is now primarily within `src.graph_utils.create_pose_graph` and its helpers,
# using the constants defined in this file.
# The ALL_CONNECTION_GROUPS is a helper, but direct iteration over specific lists like
# LEFT_FINGERS is done in graph_utils for clarity.
