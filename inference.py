#!/usr/bin/env python3
"""
Violence detection inference script for MMPose JSON files.

This script uses a trained Graph Neural Network model to predict violence
scores from human pose data in MMPose JSON format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from violence_detection_model import ViolenceDetectionGNN, create_pose_graph, get_device


def load_and_process_json(json_file: Path) -> List[Tuple[int, List[Data]]]:
    """
    Load and process a single MMPose JSON file for inference.

    Args:
        json_file: Path to the JSON file

    Returns:
        List of tuples containing (frame_id, list_of_graph_data)
    """
    graphs = []

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each frame in the JSON file
    for frame_data in data.get("instance_info", []):
        frame_id = frame_data.get("frame_id")
        instances = frame_data.get("instances", [])

        frame_graphs = []
        for instance in instances:
            keypoints = instance.get("keypoints", [])
            if keypoints:
                # Convert to numpy array
                keypoints_np = np.array(keypoints)

                # Create graph from keypoints
                graph = create_pose_graph(keypoints_np)
                if graph is not None:
                    frame_graphs.append(graph)

        if frame_graphs:
            graphs.append((frame_id, frame_graphs))

    return graphs


def predict_violence(
    model: ViolenceDetectionGNN, graphs: List[Data], device: torch.device
) -> List[float]:
    """
    Predict violence scores for graphs.

    Args:
        model: Trained GNN model
        graphs: List of graph data objects
        device: Device to run inference on

    Returns:
        List of violence scores between 0 and 1
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)

            # Add batch dimension for single graph
            if not hasattr(graph, "batch"):
                graph.batch = torch.zeros(
                    graph.x.shape[0], dtype=torch.long, device=device
                )

            # Forward pass
            score = model(graph.x, graph.edge_index, graph.batch)
            scores.append(score.item())

    return scores


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the inference script."""
    parser = argparse.ArgumentParser(description="Violence Detection from MMPose JSON")
    parser.add_argument(
        "--model_path",
        type=str,
        default="violence_detection_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input MMPose JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="violence_scores.json",
        help="Path to output JSON file",
    )
    return parser.parse_args()


def interpret_score(score: float) -> str:
    """
    Interpret a violence score as a human-readable category.

    Args:
        score: Violence score between 0 and 1

    Returns:
        String interpretation of the score
    """
    if score < 0.3:
        return "Likely non-violent"
    if score < 0.7:
        return "Ambiguous or moderate activity"
    return "Likely violent"


def main() -> None:
    """Main inference function to detect violence from pose data."""
    args = parse_arguments()

    # Convert string paths to Path objects
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    model_path = Path(args.model_path)

    # Validate input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist.")
        return

    # Use MPS if available (for Apple Silicon)
    device = get_device()
    print(f"Using device: {device}")

    # Load the model
    in_channels = 2  # X, Y coordinates
    model = ViolenceDetectionGNN(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # Load and process the input JSON file
    print(f"Processing input file: {input_file}")
    graph_data = load_and_process_json(input_file)

    if not graph_data:
        print("No valid pose data found in the input file.")
        return

    # Predict violence scores for each frame
    results = []

    for frame_id, frame_graphs in graph_data:
        frame_scores = predict_violence(model, frame_graphs, device)

        # Average the scores of all people in the frame
        avg_score = np.mean(frame_scores) if frame_scores else 0.0

        results.append(
            {
                "frame_id": frame_id,
                "violence_score": float(avg_score),
                "person_scores": [float(score) for score in frame_scores],
            }
        )

    # Calculate overall violence score
    overall_score = np.mean([r["violence_score"] for r in results]) if results else 0.0

    # Prepare output data
    output_data = {
        "file_name": input_file.name,
        "results": results,
        "overall_violence_score": float(overall_score),
    }

    # Save results to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")

    # Print overall violence score and interpretation
    if results:
        interpretation = interpret_score(overall_score)
        print(f"Overall violence score: {overall_score:.4f}")
        print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
