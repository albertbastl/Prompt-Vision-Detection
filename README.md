# Prompt Vision Detection

## Overview
This project explores techniques for detecting and localizing objects within images using natural language prompts. The goal is to build models that can take an image and a text description (e.g., "a red chair") and identify exactly where that object is located in the image, typically outputting a heatmap or bounding box.

## Key Approaches
The repository investigates two primary methods for this task:

### 1. CNN Decoder with FiLM Conditioning (NaFlex)
This approach uses a specialized Convolutional Neural Network (CNN) designed as a "tile decoder."
* **Mechanism:** It processes image feature tokens and text embeddings together.
* **FiLM Conditioning:** A key feature is the use of FiLM (Feature-wise Linear Modulation), which allows the text description to dynamically influence the visual processing layers.
* **Output:** This effectively tells the network which visual features to emphasize based on the prompt, resulting in a probability heatmap of the target object.

### 2. Open Vocabulary Contrastive Learning
This module implements a contrastive learning approach inspired by SigLIP.
* **Mechanism:** Instead of training a specific decoder for a fixed set of classes, it trains a projector to align image patch embeddings with text embeddings in a shared space.
* **Goal:** By pulling the representations of matching image-text pairs closer together and pushing non-matching ones apart, this method aims to enable "open vocabulary" detection—detecting objects based on text descriptions that the model may not have explicitly seen during training.

## Core Functionality

* **Preprocessing**
  Tools to convert raw image datasets and annotations into compressed formats (like `.npz`) containing pre-computed embeddings and training targets.

* **Training**
  Scripts to train both the CNN decoder and the contrastive projector. The training loop includes support for logging metrics (loss, accuracy, IoU) to Weights & Biases.

* **Visualization**
  A dedicated visualization tool that takes the trained model's output and overlays the generated heatmaps and bounding boxes onto the original images. This allows for easy qualitative assessment of how well the model "listens" to the text prompts.
