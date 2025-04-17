#!/usr/bin/env python3
"""
Video Pose Estimation using OpenPose
"""
import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

class PoseEstimator:
    def __init__(self, model_path=None):
        """Initialize the OpenPose model for pose estimation"""
        if model_path is None:
            model_path = Path("models")
            
        # Load the OpenPose model
        # Check if model path exists, if not create it
        if not model_path.exists():
            model_path.mkdir(parents=True)
            
        # Paths for pre-trained OpenPose models
        self.protoFile = model_path / "pose/coco/pose_deploy_linevec.prototxt"
        self.weightsFile = model_path / "pose/coco/pose_iter_440000.caffemodel"
        
        # Skip model download since we already downloaded them manually
        if not self.protoFile.exists() or not self.weightsFile.exists():
            print(f"Models not found. Please ensure the models are in the correct location:")
            print(f"  - {self.protoFile}")
            print(f"  - {self.weightsFile}")
            sys.exit(1)
            
        # Load the network
        self.net = cv2.dnn.readNetFromCaffe(str(self.protoFile), str(self.weightsFile))
        
        # COCO Output Format
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
        }
        
        # COCO colors for limbs
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
        ]
        
        # OpenPose parameters
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
    
    def process_frame(self, frame):
        """Process a single frame to detect poses"""
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        # Prepare the frame
        inputBlob = cv2.dnn.blobFromImage(
            frame, 
            1.0 / 255, 
            (self.inWidth, self.inHeight),
            (0, 0, 0), 
            swapRB=False, 
            crop=False
        )
        
        # Set the prepared input
        self.net.setInput(inputBlob)
        
        # Forward pass through the network
        output = self.net.forward()
        
        # Number of points detected
        H = output.shape[2]
        W = output.shape[3]
        
        # Empty list to store the detected keypoints
        points = []
        
        for i in range(len(self.BODY_PARTS) - 1):  # Exclude background
            # Confidence map for current body part
            probMap = output[0, i, :, :]
            
            # Find global maxima of the probMap
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit the original frame
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            
            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        # Draw the keypoints and connections
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]
            
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                
        return frame, points
    
    def process_video(self, input_path, output_path=None, display=True):
        """Process video file or camera feed and apply pose estimation"""
        # Open video or camera
        if input_path.lower() == "0" or input_path.lower() == "camera":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(input_path)
            
        if not cap.isOpened():
            print(f"Error: Could not open video source {input_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer if specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame, _ = self.process_frame(frame)
            
            # Write to output video
            if out:
                out.write(processed_frame)
                
            # Display the frame
            if display:
                cv2.imshow('Pose Estimation', processed_frame)
                
                # Progress info for video files
                if input_path.lower() != "0" and input_path.lower() != "camera":
                    print(f"Processing: {frame_idx}/{frame_count} frames", end="\r")
                    
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_idx += 1
            
        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if output_path:
            print(f"\nProcessed video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Video Pose Estimation using OpenPose")
    parser.add_argument("input", help="Path to input video file or 'camera' to use webcam")
    parser.add_argument("-o", "--output", help="Path to output video file")
    parser.add_argument("-m", "--model-path", help="Path to OpenPose model directory")
    parser.add_argument("-nd", "--no-display", action="store_true", help="Don't display output while processing")
    
    args = parser.parse_args()
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(model_path=Path(args.model_path) if args.model_path else None)
    
    # Process the video
    pose_estimator.process_video(
        args.input,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main() 