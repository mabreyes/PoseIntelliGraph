# Violence Detection from Human Pose Data

This project uses Graph Neural Networks (GNNs) to detect violent behavior from human pose estimation data. The system analyzes human poses extracted by MMPose and predicts a violence score between 0 and 1.

## Features

- Process MMPose JSON files containing pose estimation data
- Convert human pose data into graph structures
- Apply Graph Neural Networks to analyze pose interactions
- Predict violence scores on a scale from 0 to 1
- Visualize training metrics and model performance

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:

```
/path/to/violence-detection-dataset/
├── violent/
│   └── cam1/
│       └── processed/
│           ├── results_1.json
│           ├── results_2.json
│           └── ...
└── non-violent/
    └── cam1/
        └── processed/
            ├── results_1.json
            ├── results_2.json
            └── ...
```

## Usage

### Training the Model

To train the violence detection model:

```bash
python violence_detection_model.py
```

This will:
1. Load MMPose JSON files from the specified directories
2. Convert pose data to graph representations
3. Train a GNN model on the data
4. Evaluate the model performance
5. Save the trained model and performance metrics

### Making Predictions

To predict violence scores for new pose data:

```bash
python inference.py --input_file /path/to/results.json --output_file violence_scores.json
```

The output will include:
- Violence scores for each frame
- Individual person scores within each frame
- An overall violence score for the entire sequence

## Model Architecture

The violence detection model uses a Graph Convolutional Network (GCN) with:
- Multiple graph convolutional layers
- Dropout for regularization
- Global pooling for graph-level predictions
- Fully connected layers for final classification

Pose keypoints are represented as nodes in the graph, with edges connecting related body parts.

## Evaluation

The model is evaluated using:
- Binary cross-entropy loss
- ROC AUC score for classification performance
- Training and validation curves to monitor learning progress

## License

This project is licensed under the MIT License - see the LICENSE file for details.
