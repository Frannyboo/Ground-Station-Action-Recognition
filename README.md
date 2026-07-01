# Ground Station Action Recognition
Real-time human action recognition system using an X3D deep learning model for GPU-accelerated threat analysis within a distributed UAV surveillance architecture.

---

# Overview
Real-time surveillance systems require more than object detection to understand potentially suspicious activities. While object detection can identify the presence of a person, it cannot determine what that person is doing.

This project implements the ground-station action recognition module of a distributed UAV surveillance system. Instead of performing computationally intensive video analysis on the UAV, event-triggered video clips are transmitted to a GPU-equipped ground station where an X3D neural network performs temporal action recognition. The predicted action is then used to support automated threat assessment and security alert generation.

By separating lightweight object detection from computationally intensive action recognition, the system achieves efficient resource utilization while maintaining real-time surveillance capabilities.

---

# Features
* Real-time video action recognition
* X3D-based deep learning inference
* GPU-accelerated processing
* Event-driven clip analysis
* Human activity classification
* Automatic threat assessment
* Distributed AI architecture
* Integration with UAV surveillance pipeline

---
# System Pipeline

```text
Event Video Clip
        │
        ▼
Receive Video
        │
        ▼
Video Preprocessing
        │
        ▼
Frame Extraction
        │
        ▼
Clip Normalization
        │
        ▼
X3D Inference
        │
        ▼
Action Classification
        │
        ▼
Threat Assessment
        │
        ▼
Alert Generation
```

The module is designed to operate only when the edge AI detector identifies a relevant object, significantly reducing unnecessary computation and communication overhead.

---

# Hardware

| Component                  | Purpose                               |
| -------------------------- | ------------------------------------- |
| NVIDIA RTX 4060 Laptop GPU | Action recognition inference          |
| Ground Control Station     | Video processing and alert generation |

---

# Software Stack

| Technology | Purpose                        |
| ---------- | ------------------------------ |
| Python     | Primary programming language   |
| PyTorch    | Deep learning framework        |
| PySlowFast | X3D implementation             |
| OpenCV     | Video processing               |
| Flask      | Communication with edge device |

---

# Dataset

The X3D model was trained using an aerial human action recognition dataset consisting of short video clips captured from UAV perspectives.

### Dataset Characteristics

* Aerial video clips
* Human action categories
* Temporal video sequences
* Training, validation and testing subsets

Training datasets and model weights are intentionally excluded from this repository.

---

# Action Recognition Pipeline

The inference pipeline consists of the following stages:
1. Receive event-triggered video clip
2. Decode and load video frames
3. Resize and normalize frames
4. Construct fixed-length clip tensor
5. Perform X3D inference
6. Generate action probabilities
7. Select predicted action
8. Trigger security alert when required

---

# Performance

| Metric         | Result |
| -------------- | ------ |
| Top-1 Accuracy | 42.57% |
| Top-5 Accuracy | 82.51% |

The model demonstrated effective recognition of aerial human activities while maintaining inference performance suitable for real-time security monitoring on GPU hardware.

---

# Sample Results

Include screenshots demonstrating:

* Predicted action labels
* Example video clips
* Inference outputs
* Classification confidence
* Confusion matrix
* Accuracy curves

---

# Engineering Challenges
Development of the system involved addressing several practical challenges, including:
* Processing temporal video data efficiently.
* Managing GPU memory during inference.
* Integrating deep learning inference into a distributed surveillance architecture.
* Maintaining low end-to-end latency between edge detection and action classification.
* Balancing recognition accuracy with real-time performance requirements.

---

# Lessons Learned
Developing this system demonstrated that video understanding presents significantly different challenges from image-based object detection. Successful deployment required efficient preprocessing, optimized GPU inference, robust communication between distributed components, and careful consideration of latency throughout the processing pipeline. The project also highlighted the importance of modular system design, allowing each subsystem to perform tasks best suited to its available computational resources.

---

# Future Improvements
Potential future enhancements include:
* Transformer-based action recognition models
* Multi-person action recognition
* Real-time streaming inference
* Temporal action localization
* Model quantization for improved efficiency
* Support for additional action categories

---

# Repository Contents
```text
Ground-Station-Action-Recognition
│
├── README.md
├── LICENSE
├── requirements.txt
├── images/
└── src/
```

---

# Repository Structure

## images/

Contains project images used throughout this repository, including:

* System architecture
* Processing pipeline
* Prediction examples
* Confusion matrix
* Performance graphs

---

## src/

Contains the inference implementation used for action recognition, including video preprocessing, clip generation, model inference and prediction.

Training scripts, datasets and trained model weights are intentionally excluded.

---

# Related Projects

This repository represents the action recognition component of a larger distributed UAV surveillance system.

Related repositories include:

* Edge AI Aerial Object Detection
* Remote Video and Data Streaming System
* UAV Autonomous Navigation
* UAV Edge AI Surveillance System

---

# License

This project is licensed under the MIT License.

---

## Note

This repository is intended to demonstrate the deployment and inference components of the action recognition system. Training datasets, pretrained weights and selected implementation details have been intentionally omitted to protect intellectual property while showcasing the overall system architecture and engineering approach.
