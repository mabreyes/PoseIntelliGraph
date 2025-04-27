# Violence Detection from Human Pose Data

This project uses Graph Neural Networks (GNNs) to detect violent behavior from human pose estimation data. The system analyzes human poses extracted by MMPose and predicts a violence score between 0 and 1.

## Features

- Process MMPose JSON files containing pose estimation data
- Convert human pose data into graph structures
- Apply Graph Neural Networks to analyze pose interactions
- Predict violence scores on a scale from 0 to 1
- Visualize training metrics and model performance
- Hardware acceleration support (CUDA, MPS for Apple Silicon)

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

#### Basic Training Command

```bash
python violence_detection_model.py
```

#### Modifying Training Parameters

Before running training, you can adjust these parameters in `violence_detection_model.py`:

```python
# Constants
DATA_PATH = "/path/to/violence-detection-dataset"  # Change to your dataset path
VIOLENT_PATH = os.path.join(DATA_PATH, "violent/cam1/processed")
NON_VIOLENT_PATH = os.path.join(DATA_PATH, "non-violent/cam1/processed")
BATCH_SIZE = 32  # Adjust based on your memory constraints
NUM_EPOCHS = 50  # Increase for better performance, decrease for faster testing
LEARNING_RATE = 0.001  # Adjust if needed
```

#### Training Examples

1. **Quick Test Training (1 epoch)**

   Edit `violence_detection_model.py` to set `NUM_EPOCHS = 1`, then run:
   ```bash
   python violence_detection_model.py
   ```

2. **Full Training (50 epochs)**

   Edit `violence_detection_model.py` to set `NUM_EPOCHS = 50`, then run:
   ```bash
   python violence_detection_model.py
   ```

3. **Training with Specific Batch Size**

   Edit `violence_detection_model.py` to set `BATCH_SIZE = 16` (for lower memory usage), then run:
   ```bash
   python violence_detection_model.py
   ```

The training script automatically uses hardware acceleration when available:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon (M1/M2/M3 chips)
- CPU for other systems

#### Training Output

Upon successful training, you'll see:
```
Using device: mps  # Or cuda/cpu depending on your hardware
Loading and preprocessing data...
Found 44 violent JSON files
Processing violent samples: 100%|██████████| 44/44 [00:35<00:00, 1.23it/s]
Total graphs: 528
Positive (violent) samples: 264.0
Negative (non-violent) samples: 264.0
Training graphs: 316
Validation graphs: 106
Test graphs: 106
Training model...
Epoch 1/1:
  Train Loss: 0.6931
  Val Loss: 0.6931
  Val AUC: 0.5043
Test Loss: 0.6931
Test AUC: 0.5021
Model saved to violence_detection_model.pt
Training metrics plot saved to training_metrics.png
```

The training process:
1. Loads MMPose JSON files from both violent and non-violent datasets
2. Converts pose data to graph representations
3. Trains a GNN model on the data
4. Evaluates the model performance
5. Saves the trained model to `violence_detection_model.pt`
6. Generates training metrics visualization in `training_metrics.png`

### Making Predictions

#### Basic Inference Command

```bash
python inference.py --input_file /path/to/results.json --output_file violence_scores.json
```

#### Command-line Arguments

The inference script accepts the following arguments:

- `--input_file`: Path to the MMPose JSON file (required)
- `--output_file`: Path to save the output results (default: `violence_scores.json`)
- `--model_path`: Path to the trained model (default: `violence_detection_model.pt`)

#### Inference Examples

1. **Basic Inference**

   ```bash
   python inference.py --input_file /Volumes/MARCREYES/violence-detection-dataset/violent/cam1/processed/results_1.json
   ```

2. **Inference with Custom Output File**

   ```bash
   python inference.py --input_file /Volumes/MARCREYES/violence-detection-dataset/violent/cam1/processed/results_1.json --output_file results_1_scores.json
   ```

3. **Inference with Custom Model**

   ```bash
   python inference.py --input_file /Volumes/MARCREYES/violence-detection-dataset/violent/cam1/processed/results_1.json --model_path custom_model.pt
   ```

4. **Analyzing Multiple Files in Sequence**

   ```bash
   # Analyze file 1
   python inference.py --input_file /Volumes/MARCREYES/violence-detection-dataset/violent/cam1/processed/results_1.json --output_file scores_1.json

   # Analyze file 2
   python inference.py --input_file /Volumes/MARCREYES/violence-detection-dataset/violent/cam1/processed/results_2.json --output_file scores_2.json
   ```

#### Inference Output

The script will produce output like:
```
Using device: mps
Model loaded from violence_detection_model.pt
Processing input file: /Volumes/MARCREYES/violence-detection-dataset/violent/cam1/processed/results_1.json
Results saved to violence_scores.json
Overall violence score: 0.9842
Interpretation: Likely violent
```

The output JSON file will have this structure:
```json
{
  "file_name": "results_1.json",
  "results": [
    {
      "frame_id": 1,
      "violence_score": 0.9785136580467224,
      "person_scores": [
        0.9372978806495667,
        0.9982445240020752,
        0.9999985694885254
      ]
    },
    ...
  ],
  "overall_violence_score": 0.9842
}
```

#### Interpretation of Scores

The model provides these interpretations based on the overall violence score:
- Below 0.3: "Likely non-violent"
- Between 0.3 and 0.7: "Ambiguous or moderate activity"
- Above 0.7: "Likely violent"

## Model Architecture

The violence detection model uses a Graph Convolutional Network (GCN) with:
- Multiple graph convolutional layers
- Dropout for regularization
- Global pooling for graph-level predictions
- Fully connected layers for final classification

Pose keypoints are represented as nodes in the graph, with edges connecting related body parts.

## Performance

On violent pose sequences, the model typically predicts scores above 0.98, correctly identifying violent behavior. The model's performance is evaluated using:
- Binary cross-entropy loss
- ROC AUC score for classification performance
- Training and validation curves to monitor learning progress

## License

This project is licensed under the MIT License - see the LICENSE file for details.
