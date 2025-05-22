"""
Utilities for graph creation and manipulation in the Violence Detection GNN project.

This module includes functions for generating pose graphs from keypoint data
and loading data from MMPose JSON files.
"""

# Standard library imports
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Third-party imports
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Import specific connection lists and constants
from .pose_connections import (
    BODY_CONNECTIONS,
    FACE_TO_BODY_CONNECTIONS,
    LEFT_HAND_TO_BODY,
    RIGHT_HAND_TO_BODY,
    LEFT_FINGERS, # This is a list of lists (each list is a finger)
    RIGHT_FINGERS, # This is a list of lists
    LEFT_FINGER_BASES, # List of keypoint indices for bases of left fingers
    RIGHT_FINGER_BASES, # List of keypoint indices for bases of right fingers
    FACE_KEYPOINTS_START_IDX,
    FACE_KEYPOINTS_END_IDX,
    FACE_CENTER_POINT_IDX,
    MIN_CONFIDENCE_THRESHOLD
)

def _add_edges_from_connection_list(
    valid_keypoints_data: np.ndarray, # Should contain x,y coordinates at least
    connections: List[Tuple[int, int]],
    index_map: Dict[int, int],
    edge_list: List[List[int]],
    edge_features: List[List[float]],
    add_edge_attr: bool,
) -> None:
    """Helper function to add edges based on a list of connections."""
    for src_orig, dst_orig in connections:
        if src_orig in index_map and dst_orig in index_map:
            src_mapped, dst_mapped = index_map[src_orig], index_map[dst_orig]
            
            # Avoid duplicate edges (though PyG handles this, explicit check is cleaner)
            if [src_mapped, dst_mapped] not in edge_list:
                edge_list.append([src_mapped, dst_mapped])
                edge_list.append([dst_mapped, src_mapped]) # Bidirectional

                if add_edge_attr:
                    # Use only x,y for distance calculation (first 2 elements)
                    dist = np.linalg.norm(
                        valid_keypoints_data[src_mapped, :2] - valid_keypoints_data[dst_mapped, :2]
                    )
                    edge_features.append([dist])
                    edge_features.append([dist])

def _connect_sequential_keypoints(
    valid_keypoints_data: np.ndarray,
    keypoint_indices_orig: List[int], # List of original keypoint indices in sequence
    index_map: Dict[int, int],
    edge_list: List[List[int]],
    edge_features: List[List[float]],
    add_edge_attr: bool,
    closed_loop: bool = False
) -> None:
    """Connects a list of keypoints sequentially."""
    mapped_indices = [index_map[i] for i in keypoint_indices_orig if i in index_map]
    if len(mapped_indices) < 2:
        return

    for i in range(len(mapped_indices) - 1):
        src_mapped, dst_mapped = mapped_indices[i], mapped_indices[i+1]
        if [src_mapped, dst_mapped] not in edge_list:
            edge_list.extend([[src_mapped, dst_mapped], [dst_mapped, src_mapped]])
            if add_edge_attr:
                dist = np.linalg.norm(valid_keypoints_data[src_mapped, :2] - valid_keypoints_data[dst_mapped, :2])
                edge_features.extend([[dist], [dist]])
    
    if closed_loop and len(mapped_indices) > 2: # Connect last to first
        src_mapped, dst_mapped = mapped_indices[-1], mapped_indices[0]
        if [src_mapped, dst_mapped] not in edge_list:
            edge_list.extend([[src_mapped, dst_mapped], [dst_mapped, src_mapped]])
            if add_edge_attr:
                dist = np.linalg.norm(valid_keypoints_data[src_mapped, :2] - valid_keypoints_data[dst_mapped, :2])
                edge_features.extend([[dist], [dist]])


def _connect_points_to_center(
    valid_keypoints_data: np.ndarray,
    point_indices_orig: List[int],
    center_point_orig_idx: int,
    index_map: Dict[int, int],
    edge_list: List[List[int]],
    edge_features: List[List[float]],
    add_edge_attr: bool,
) -> None:
    """Connects a list of points to a central point."""
    if center_point_orig_idx not in index_map:
        return
    
    center_mapped_idx = index_map[center_point_orig_idx]
    mapped_point_indices = [index_map[i] for i in point_indices_orig if i in index_map]

    for point_mapped_idx in mapped_point_indices:
        if center_mapped_idx != point_mapped_idx: # Avoid self-loops
            if [center_mapped_idx, point_mapped_idx] not in edge_list:
                edge_list.extend([[center_mapped_idx, point_mapped_idx], [point_mapped_idx, center_mapped_idx]])
                if add_edge_attr:
                    dist = np.linalg.norm(valid_keypoints_data[center_mapped_idx, :2] - valid_keypoints_data[point_mapped_idx, :2])
                    edge_features.extend([[dist], [dist]])

def create_pose_graph(keypoints_with_conf: np.ndarray, edge_attr: bool = True) -> Optional[Data]:
    """
    Convert keypoints (x, y, confidence) into a graph representation.
    Uses anatomical connections defined in pose_connections.py.

    Args:
        keypoints_with_conf: NumPy array of shape (num_keypoints, 3)
                             containing (x, y, confidence_score).
        edge_attr: Whether to include edge attributes (distances).

    Returns:
        PyTorch Geometric Data object or None if not enough valid keypoints.
    """
    # Filter keypoints by confidence and validity (non-zero, non-NaN)
    valid_conf_mask = keypoints_with_conf[:, 2] > MIN_CONFIDENCE_THRESHOLD
    non_zero_mask = (keypoints_with_conf[:, :2] != 0).any(axis=1)
    nan_mask = ~np.isnan(keypoints_with_conf[:, :2]).any(axis=1)
    final_valid_mask = valid_conf_mask & non_zero_mask & nan_mask
    
    valid_keypoints_data = keypoints_with_conf[final_valid_mask]

    if len(valid_keypoints_data) < 2: # Need at least 2 points for an edge. Original had <3.
        return None

    # Node features are the 2D coordinates (x, y)
    # valid_keypoints_data still contains all 3 columns (x, y, conf) for other uses (like distance calc)
    # But the 'x' tensor for node features should only have x, y.
    x = torch.tensor(valid_keypoints_data[:, :2], dtype=torch.float)
    num_nodes = len(valid_keypoints_data)

    original_indices = np.where(final_valid_mask)[0]
    index_map: Dict[int, int] = {orig_idx: new_idx for new_idx, orig_idx in enumerate(original_indices)}

    edge_list: List[List[int]] = []
    edge_features: List[List[float]] = []

    # 1. Standard Body Connections
    _add_edges_from_connection_list(valid_keypoints_data, BODY_CONNECTIONS, index_map, edge_list, edge_features, edge_attr)

    # 2. Face Connections
    #    a. Sequentially connect contour points (original behavior for face_indices)
    face_contour_orig_indices = list(range(FACE_KEYPOINTS_START_IDX, FACE_KEYPOINTS_END_IDX + 1))
    _connect_sequential_keypoints(valid_keypoints_data, face_contour_orig_indices, index_map, edge_list, edge_features, edge_attr, closed_loop=False)
    #    b. Connect all face contour points to the main face center point (e.g., nose)
    _connect_points_to_center(valid_keypoints_data, face_contour_orig_indices, FACE_CENTER_POINT_IDX, index_map, edge_list, edge_features, edge_attr)
    #    c. Connect face to body (e.g. nose to first face contour point)
    _add_edges_from_connection_list(valid_keypoints_data, FACE_TO_BODY_CONNECTIONS, index_map, edge_list, edge_features, edge_attr)


    # 3. Hand Connections
    #    a. Connect COCO wrist to Hand Root (e.g. COCO L-Wrist to L-Hand-Root)
    _add_edges_from_connection_list(valid_keypoints_data, LEFT_HAND_TO_BODY, index_map, edge_list, edge_features, edge_attr)
    _add_edges_from_connection_list(valid_keypoints_data, RIGHT_HAND_TO_BODY, index_map, edge_list, edge_features, edge_attr)
    
    #    b. Connect joints within each finger for left hand
    for finger_connections_orig in LEFT_FINGERS: # each item is a list of joints for one finger
        _connect_sequential_keypoints(valid_keypoints_data, finger_connections_orig, index_map, edge_list, edge_features, edge_attr)
    
    #    c. Connect joints within each finger for right hand
    for finger_connections_orig in RIGHT_FINGERS:
        _connect_sequential_keypoints(valid_keypoints_data, finger_connections_orig, index_map, edge_list, edge_features, edge_attr)

    #    d. Connect finger bases sequentially (e.g., thumb base to index base, index to middle, etc.)
    _connect_sequential_keypoints(valid_keypoints_data, LEFT_FINGER_BASES, index_map, edge_list, edge_features, edge_attr, closed_loop=False) # Not a closed loop for bases
    _connect_sequential_keypoints(valid_keypoints_data, RIGHT_FINGER_BASES, index_map, edge_list, edge_features, edge_attr, closed_loop=False)

    # 4. Ensure graph connectivity (fallback if too sparse or no edges)
    #    Original logic: if not edge_list or len(edge_list) < 2 * num_nodes: connect all.
    #    This can make graph very dense.
    #    If graph is too sparse (e.g., fewer edges than nodes), add more connections.
    #    This is a simplified fallback to ensure some connectivity.
    if num_nodes > 1 and (not edge_list or len(edge_list) < num_nodes):
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if [i, j] not in edge_list and [j, i] not in edge_list:
                    edge_list.extend([[i,j],[j,i]])
                    if edge_attr:
                        dist = np.linalg.norm(valid_keypoints_data[i, :2] - valid_keypoints_data[j, :2])
                        edge_features.extend([[dist],[dist]])
    
    if not edge_list:
        if num_nodes > 1 : # if still no edges and more than one node, connect first two
            edge_list.extend([[0,1],[1,0]])
            if edge_attr:
                dist = np.linalg.norm(valid_keypoints_data[0,:2] - valid_keypoints_data[1,:2])
                edge_features.extend([[dist],[dist]])
        else: # Single node graph or no valid edges possible
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0,1), dtype=torch.float) if edge_attr else None
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if edge_attr and edge_features:
            edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)
        elif edge_attr : # features list might be empty if all edges were added by fallback
             edge_attr_tensor = torch.empty((edge_index.shape[1],1), dtype=torch.float) # Create dummy if needed
        else:
            edge_attr_tensor = None
            
    data = Data(x=x, edge_index=edge_index)
    if edge_attr_tensor is not None and edge_attr_tensor.shape[0] == edge_index.shape[1]:
        data.edge_attr = edge_attr_tensor
    elif edge_attr and edge_index.shape[1] > 0:
        # If edge_attr was true, but tensor is mismatched or None,
        # and there are edges, fill with a default value (e.g., 1.0).
        default_edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
        data.edge_attr = default_edge_attr
    return data


def load_mmpose_data(
    violent_path: Path, non_violent_path: Path, sample_percentage: int = 100
) -> Tuple[List[Data], List[float]]:
    """
    Load MMPose JSON files and convert them to graph data.

    Processes JSON files containing pose keypoints from both violent and non-violent
    video frames. Each person instance in a frame is converted to a graph representation
    suitable for GNN processing. The function supports processing a subset of the data
    using the sample_percentage parameter.

    Args:
        violent_path: Path to violent pose JSON files
        non_violent_path: Path to non-violent pose JSON files
        sample_percentage: Percentage of files to process (1-100)

    Returns:
        Tuple of (list of graph Data objects, list of corresponding labels)
    """
    # Validate sample percentage
    if not 1 <= sample_percentage <= 100:
        raise ValueError("sample_percentage must be between 1 and 100")

    all_graphs = []
    all_labels = []

    # Get all JSON files from the violent directory
    violent_files = list(violent_path.glob("*.json"))
    if not violent_files:
        raise ValueError(f"No JSON files found in violent directory: {violent_path}")

    print(f"Found {len(violent_files)} violent JSON files")

    # Calculate number of files to process based on percentage
    num_violent_files = max(1, int(len(violent_files) * sample_percentage / 100))

    # Process violent samples
    for json_file in tqdm(
        violent_files[:num_violent_files], desc="Processing violent samples"
    ):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process each frame in the JSON file
        for frame_data in data.get("instance_info", []):
            # Get frame ID (not used but kept for consistency)
            _ = frame_data.get("frame_id")
            instances = frame_data.get("instances", [])

            for instance in instances:
                keypoints = instance.get("keypoints", [])
                # Ensure keypoints are in (N, 3) format (x, y, conf)
                # Original data from MMPose might be (N_person, N_keypoints, 2_or_3)
                # This loader expects keypoints for a single person instance to be (N_keypoints, 3)
                if keypoints:
                    keypoints_np = np.array(keypoints)
                    if keypoints_np.ndim == 2 and keypoints_np.shape[1] == 2: # only x,y
                        # Add a dummy confidence of 1.0 if not present
                        keypoints_np = np.hstack([keypoints_np, np.ones((keypoints_np.shape[0], 1))])
                    
                    # At this point, keypoints_np should be (num_total_keypoints, 3)
                    # create_pose_graph handles filtering by confidence internally
                    graph = create_pose_graph(keypoints_np) # Pass the (N,3) array
                    if graph is not None:
                        all_graphs.append(graph)
                        all_labels.append(1.0)  # Violent label

    # Process non-violent data
    non_violent_files = list(non_violent_path.glob("*.json"))
    if not non_violent_files:
        raise ValueError(
            f"No JSON files found in non-violent directory: {non_violent_path}"
        )

    print(f"Found {len(non_violent_files)} non-violent JSON files")

    # Calculate number of files to process based on percentage
    num_nonviolent_files = max(1, int(len(non_violent_files) * sample_percentage / 100))

    # Process non-violent samples
    for json_file in tqdm(
        non_violent_files[:num_nonviolent_files], desc="Processing non-violent samples"
    ):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process each frame in the JSON file
        for frame_data in data.get("instance_info", []):
            # Get frame ID (not used but kept for consistency)
            _ = frame_data.get("frame_id")
            instances = frame_data.get("instances", [])

            for instance in instances:
                keypoints = instance.get("keypoints", [])
                if keypoints:
                    keypoints_np = np.array(keypoints)
                    if keypoints_np.ndim == 2 and keypoints_np.shape[1] == 2: # only x,y
                        keypoints_np = np.hstack([keypoints_np, np.ones((keypoints_np.shape[0], 1))])

                    graph = create_pose_graph(keypoints_np)
                    if graph is not None:
                        all_graphs.append(graph)
                        all_labels.append(0.0)  # Non-violent label
    return all_graphs, all_labels
