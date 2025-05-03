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
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch_geometric.data import Data

# Import from separate component files
from gnn import create_pose_graph
from violence_detection_model import ViolenceDetectionGNN, get_device


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Violence classification threshold (0.0-1.0). If not provided, uses threshold from model file",
    )
    parser.add_argument(
        "--show_metrics", 
        action="store_true",
        help="Show threshold metrics from the model"
    )
    return parser.parse_args()


def load_model_and_threshold(model_path: Path, device: torch.device) -> Tuple[ViolenceDetectionGNN, float, Optional[Dict]]:
    """
    Load the model and threshold from a saved model file.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model to
        
    Returns:
        Tuple of (model, threshold, metrics)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    in_channels = 2
    model = ViolenceDetectionGNN(
        in_channels=in_channels,
        hidden_channels=64,
        transformer_heads=4,
        transformer_layers=2
    ).to(device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        threshold = checkpoint.get('threshold', 0.5)
        metrics = checkpoint.get('metrics', None)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5
        metrics = None
    
    return model, threshold, metrics


def interpret_score(score: float, threshold: float) -> Tuple[str, bool]:
    """
    Interpret a violence score based on the threshold.

    Args:
        score: Violence score between 0 and 1
        threshold: Classification threshold

    Returns:
        Tuple of (interpretation string, is_violent boolean)
    """
    is_violent = score >= threshold
    
    if score < threshold - 0.2:
        return "Likely non-violent", is_violent
    elif score < threshold:
        return "Possibly non-violent", is_violent
    elif score < threshold + 0.2:
        return "Possibly violent", is_violent
    else:
        return "Likely violent", is_violent


def main() -> None:
    """Main inference function to detect violence from pose data."""
    args = parse_arguments()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    model_path = Path(args.model_path)

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist.")
        return

    device = get_device()
    print(f"Using device: {device}")

    model, model_threshold, metrics = load_model_and_threshold(model_path, device)
    print(f"Model loaded from {model_path}")
    
    threshold = args.threshold if args.threshold is not None else model_threshold
    print(f"Using classification threshold: {threshold}" + 
          (" (from model)" if args.threshold is None else " (user-specified)"))
    
    if args.show_metrics and metrics:
        print("\nModel threshold metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print(f"Processing input file: {input_file}")
    graph_data = load_and_process_json(input_file)

    if not graph_data:
        print("No valid pose data found in the input file.")
        return

    results = []
    violent_frame_count = 0

    for frame_id, frame_graphs in graph_data:
        frame_scores = predict_violence(model, frame_graphs, device)
        avg_score = np.mean(frame_scores) if frame_scores else 0.0
        
        interpretation, is_violent = interpret_score(avg_score, threshold)
        if is_violent:
            violent_frame_count += 1

        results.append({
            "frame_id": frame_id,
            "violence_score": float(avg_score),
            "is_violent": bool(is_violent),
            "interpretation": interpretation,
            "person_scores": [float(score) for score in frame_scores],
        })

    overall_score = np.mean([r["violence_score"] for r in results]) if results else 0.0
    overall_interpretation, is_violent_overall = interpret_score(overall_score, threshold)
    
    violent_percentage = (violent_frame_count / len(results)) * 100 if results else 0

    output_data = {
        "file_name": str(input_file.name),
        "results": results,
        "overall_violence_score": float(overall_score),
        "is_violent_overall": bool(is_violent_overall),
        "violent_frame_percentage": float(violent_percentage),
        "classification_threshold": float(threshold),
        "interpretation": str(overall_interpretation)
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Overall violence score: {overall_score}")
    print(f"Interpretation: {overall_interpretation}")
    print(f"Violent frames: {violent_frame_count}/{len(results)} ({violent_percentage}%)")


if __name__ == "__main__":
    main()
