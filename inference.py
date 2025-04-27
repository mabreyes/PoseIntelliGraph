import argparse
import json
import os

import numpy as np
import torch

from violence_detection_model import ViolenceDetectionGNN, create_pose_graph, get_device


def load_and_process_json(json_file):
    """
    Load and process a single MMPose JSON file for inference.

    Args:
        json_file: Path to the JSON file

    Returns:
        List of graph data objects
    """
    graphs = []

    with open(json_file, "r") as f:
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


def predict_violence(model, graphs, device):
    """
    Predict violence scores for graphs.

    Args:
        model: Trained GNN model
        graphs: List of graph data objects
        device: Device to run inference on

    Returns:
        List of violence scores
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


def main():
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
    args = parser.parse_args()

    # Use MPS if available (for Apple Silicon)
    device = get_device()
    print(f"Using device: {device}")

    # Load the model
    in_channels = 2  # X, Y coordinates
    model = ViolenceDetectionGNN(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    # Load and process the input JSON file
    print(f"Processing input file: {args.input_file}")
    graph_data = load_and_process_json(args.input_file)

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
                "violence_score": avg_score,
                "person_scores": frame_scores,
            }
        )

    # Save results to output file
    with open(args.output_file, "w") as f:
        json.dump(
            {
                "file_name": os.path.basename(args.input_file),
                "results": results,
                "overall_violence_score": np.mean(
                    [r["violence_score"] for r in results]
                )
                if results
                else 0.0,
            },
            f,
            indent=2,
        )

    print(f"Results saved to {args.output_file}")

    # Print overall violence score
    if results:
        overall_score = np.mean([r["violence_score"] for r in results])
        print(f"Overall violence score: {overall_score:.4f}")

        # Interpret the score
        if overall_score < 0.3:
            print("Interpretation: Likely non-violent")
        elif overall_score < 0.7:
            print("Interpretation: Ambiguous or moderate activity")
        else:
            print("Interpretation: Likely violent")


if __name__ == "__main__":
    main()
