# 3D Object Detection on RGB-D Data: Technical Documentation

## Project Overview

This project tackles 3D object detection using multimodal RGB-D data consisting of 200 samples. Given the limited dataset size, I explored two distinct approaches:

1. **Training a custom architecture from scratch**, allowing full control over design and optimization.
2. **Fine-tuning a state-of-the-art pre-trained model** to benefit from transfer learning and better generalization.

👉 **Interactive Overview**: [📊 3D Pipeline Visualization (Rendered)](https://hamidthri.github.io/3D_Bbox_detection/3d_pipeline_visualization.html)

These resources provide visual and written explanations of the architecture, data flow, and design rationale.



### Approach 1: Custom Training from Scratch

The second approach involved developing a custom multimodal architecture from the ground up. While this provided greater control over the model design and training process, it presented significant challenges given the small dataset size (200 samples), making it particularly difficult to achieve robust generalization without overfitting.

## Data Preprocessing

The dataset consists of RGB images, point clouds, 3D bounding box corners, and instance segmentation masks, stored in folders under `data/dl_challenge`. To prepare this data for training, I implemented a custom `BBox3DDataset` class that handles loading, augmentation, and transformation into a format suitable for the neural network.

### Image Preprocessing
RGB images are processed with the following steps:
- **Resizing**: Images are resized to a consistent resolution of 480×608 pixels to ensure uniformity across the dataset.
- **Training Augmentations**: For the training split, I apply the following augmentations to improve model robustness while preserving spatial relationships:
  - **Random Horizontal Flip**: Applied with a 50% probability to simulate objects viewed from different angles.
  - **Random Rotation**: Small rotations up to ±10 degrees to mimic slight camera tilts.
  - **Color Jitter**: Random adjustments to brightness, contrast, and saturation (each with a factor of 0.2) to handle varying lighting conditions.
  - **Random Resized Crop**: Crops and resizes images with a scale range of 0.8–1.0 and an aspect ratio of 0.9–1.1 to simulate different framings or distances.
- **Normalization**: Images are normalized using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` to match the pretrained EfficientNet-B3 backbone.
- For validation, only resizing and normalization are applied to ensure consistent evaluation.

### Point Cloud Preprocessing
Point clouds are provided as 3D arrays and processed as follows:
- **Reshaping**: The point cloud is reshaped from `(3, H, W)` to `(H*W, 3)` to represent a set of 3D points.
- **Filtering**: Invalid points (NaN values or negative z-coordinates) are removed to ensure only valid 3D points are used.
- **Sampling/Padding**: To maintain a consistent input size, point clouds are either randomly downsampled to 8192 points (if larger) or padded with zeros (if smaller).

### Bounding Box Representation Conversion
The original dataset provides 3D bounding boxes as 8-corner coordinates `(N, 8, 3)`. To simplify neural network predictions and improve training stability, I convert these to a parametric representation consisting of:
- **Center**: The 3D coordinates of the box’s center `(x, y, z)`.
- **Size**: The box’s dimensions `(width, height, depth)`.
- **Rotation Quaternion**: A 4D quaternion representing the box’s orientation.

This conversion is implemented in the `BBoxCornerToParametric` class, which uses PCA-based fitting to compute the center, size, and rotation from the corner coordinates. The resulting parameters are concatenated into a `(N, 10)` tensor per sample.

**Benefits of this conversion**:
- **Reduced Dimensionality**: Predicting 10 parameters (center, size, quaternion) instead of 24 coordinates (8 corners × 3) simplifies the regression task.
- **Regression-Friendly**: Parametric representation is more intuitive for L1 loss and avoids the need for post-processing to extract box properties.
- **Numerical Stability**: The PCA-based approach handles varying box orientations and sizes robustly, with safeguards for edge cases (e.g., degenerate boxes).

**Implementation Details**:
- The conversion is performed in `convert_corners_to_params_tensor`, which processes batches of corner data and handles invalid cases by returning default values (e.g., zero center, small size, identity quaternion).
- During training, bounding box parameters are padded or truncated to a maximum of 21 objects per sample to ensure consistent tensor shapes.

### Mask Preprocessing
Instance segmentation masks are provided as `(N, H, W)` arrays. Each mask is resized to the target image size (480×608) using nearest-neighbor interpolation to preserve binary-like values. Masks are padded or truncated to match the maximum number of objects (21) and converted to float32 tensors.

This preprocessing ensures that all inputs—RGB images, point clouds, bounding box parameters, and masks—are consistently formatted and augmented appropriately for training and validation.


## Architecture Exploration

To process the point cloud data effectively, I evaluated several state-of-the-art architectures, focusing on their computational efficiency, feature extraction quality, and suitability for a small dataset (200 samples):

- **PointNet++**: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" - [Original Paper](https://arxiv.org/abs/1706.02413) | [Primary Repository](https://github.com/charlesq34/pointnet2)
- **DGCNN**: "Dynamic Graph CNN for Learning on Point Clouds" - [Original Paper](https://arxiv.org/abs/1801.07829) | [Primary Repository](https://github.com/WangYueFt/dgcnn)
- **Sparse Convolutional Networks (Minkowski Engine)**: "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks" - [Original Paper](https://arxiv.org/abs/1904.08755) | [Primary Repository](https://github.com/NVIDIA/MinkowskiEngine)
- **PointPillars**: "PointPillars: Fast Encoders for Object Detection from Point Clouds" - [Original Paper](https://arxiv.org/abs/1812.05784) | [Primary Repository](https://github.com/nutonomy/second.pytorch)

After thorough evaluation, **DGCNN** was selected as the optimal backbone for point cloud processing. Its dynamic graph construction, which builds k-nearest neighbor graphs (k=20) to capture local geometric relationships, provides robust feature extraction while remaining computationally efficient. This makes it particularly well-suited for our limited dataset, where overfitting is a concern.

## Final Architecture

The implemented solution, embodied in the `BBox3DPredictor` class, follows a multimodal fusion approach to combine RGB images and point clouds for 3D bounding box prediction. The architecture is designed to balance performance and efficiency, with a total parameter count well below the 100M limit.

### RGB Processing
- **Backbone**: Pre-trained EfficientNet-B3, initialized with ImageNet weights (not frozen, allowing fine-tuning for our task).
- **Feature Extraction**: Extracts 1000-dimensional features, which are projected to a 512-dimensional representation using a linear layer, ReLU activation, and dropout (0.2) for robustness.
- **Augmentation**: For training, applies color-space and spatial transformations that preserve 3D spatial integrity:
  - Random Horizontal Flip (p=0.5)
  - Random Rotation (±10 degrees)
  - Color Jitter (brightness, contrast, saturation with factor 0.2)
  - Random Resized Crop (scale 0.8–1.0, aspect ratio 0.9–1.1)
  - Normalization using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

### Point Cloud Processing
- **Architecture**: DGCNN-based feature extractor with three EdgeConv layers, each using k=20 nearest neighbors to build dynamic graphs.
- **Processing**: Takes point clouds of 8192 points (3D coordinates) and produces 256-dimensional global features through a series of graph convolutions, batch normalization, and max-pooling.
- **Output**: A compact 512-dimensional feature vector after a final linear layer, batch normalization, and dropout (0.5).

### Multimodal Fusion
- **Method**: Transformer-based attention mechanism to integrate RGB and point cloud features.
- **Layers**: 4 transformer encoder layers, each with 8 attention heads and a feedforward dimension of 2048, using a feature dimension of 512.
- **Integration**: Late fusion, where RGB and point cloud features are stacked and processed by the transformer, then averaged to produce a unified 512-dimensional feature vector.

### Prediction Heads
- **Bounding Box Head**: Predicts parameters (center, size, quaternion) for up to 21 objects per scene, outputting a tensor of shape `(batch_size, 21, 10)`. The head consists of three linear layers (512→512, 512→256, 256→210) with ReLU activations and dropout (0.2).
- **Confidence Head**: Predicts per-object detection confidence scores, outputting a tensor of shape `(batch_size, 21)` with sigmoid activation for scores between 0 and 1. The head uses two linear layers (512→256, 256→21) with ReLU and dropout (0.2).

This architecture leverages the strengths of EfficientNet-B3 for image processing, DGCNN for point cloud processing, and a transformer for robust feature fusion, ensuring accurate 3D bounding box predictions while remaining computationally efficient.

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
```bash
# Download dataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=11s-GLb6LZ0SCAVW6aikqImuuQEEbT_Fb' -O dataset.tar

# Extract data
mkdir -p data
tar -xf dataset.tar -C data/

# Verify data structure
ls data/dl_challenge/
```

### Training
```bash
python main.py
```

## Key Challenges and Solutions

1. **Limited Dataset Size**: With only 200 samples, preventing overfitting was paramount. Solutions included aggressive dropout, careful augmentation, and pre-trained feature extractors.

2. **Multimodal Registration**: Ensuring RGB and point cloud features remain aligned after preprocessing required careful coordinate system management.

3. **Parametric Conversion Stability**: Developing robust algorithms for corner-to-parametric conversion that handle degenerate cases and maintain differentiability.

4. **Memory Constraints**: Efficiently processing high-resolution RGB images alongside dense point clouds within GPU memory limits.

The small dataset size ultimately made training from scratch particularly challenging, reinforcing the value of transfer learning approaches for limited-data scenarios in 3D computer vision tasks.

## Loss Function Design and Challenges

### Core Challenge: Set Prediction Problem

3D object detection presents a fundamental challenge: predicting a variable number of objects without knowing their correspondence to ground truth targets. Unlike traditional computer vision tasks with fixed outputs, our model must predict up to 21 objects per scene while handling cases where fewer objects are present.

### Hungarian Matching Strategy

To address the correspondence problem, I implemented a Hungarian matching algorithm that optimally assigns predicted bounding boxes to ground truth targets based on a composite cost function:

```python
# Cost combines geometric similarity across all bbox parameters
cost = w_center * center_distance + w_size * size_distance + w_rot * rotation_distance
```

This matching strategy ensures that each ground truth object is assigned to exactly one prediction, eliminating ambiguity in loss computation and preventing the model from converging to degenerate solutions.

### Multi-Component Loss Design

The final loss function addresses four critical aspects of 3D object detection:

#### 1. **Center Regression Loss** (L1)
```python
center_loss = F.l1_loss(predicted_centers, ground_truth_centers)
```
- **Rationale**: L1 loss provides robust gradients for spatial localization
- **Benefit**: Less sensitive to outliers compared to L2, crucial for limited training data

#### 2. **Size Regression Loss** (L1)
```python
size_loss = F.l1_loss(predicted_sizes, ground_truth_sizes)
```
- **Challenge**: Objects vary significantly in scale (small screws vs. large components)
- **Solution**: L1 loss handles scale variations better than MSE

#### 3. **Rotation Loss** (Quaternion Cosine Distance)
```python
rotation_loss = 1.0 - torch.abs((pred_quat * gt_quat).sum(dim=-1))
```
- **Challenge**: Quaternion space has unique properties (double-cover, normalization requirements)
- **Solution**: Cosine distance respects quaternion geometry and handles the double-cover problem
- **Alternative considered**: Geodesic distance was too computationally expensive for training

#### 4. **Confidence Loss** (Focal Loss Variant)
```python
# Weighted BCE to handle class imbalance
conf_loss = alpha * positive_loss + (1-alpha) * negative_loss
```
- **Challenge**: Severe class imbalance (most predictions should be "no object")
- **Solution**: Up-weighting positive samples (α=0.75) to prevent model collapse to "predict nothing"

### Loss Balancing Strategy

Each loss component operates on different scales and units:
- Center/Size losses: Euclidean distances (meters)
- Rotation loss: Angular similarity [0,1]
- Confidence loss: Probability space [0,1]

**Weighting strategy**:
```python
total_loss = w_center * center_loss + w_size * size_loss + 
             w_rot * rotation_loss + w_conf * confidence_loss
```

Initial weights were set to unity, then adjusted during training based on gradient magnitudes to ensure balanced optimization across all components.

### Key Design Decisions

1. **Why Hungarian Matching?**: Alternative approaches like nearest-neighbor assignment led to unstable training due to inconsistent correspondences between epochs.

2. **Why L1 over L2?**: With limited data, L2 loss amplified the impact of outliers, causing training instability. L1 provided more robust gradients.

3. **Why Custom Rotation Loss?**: Standard geodesic distance on SO(3) was computationally prohibitive. The cosine distance approximation provided 90% of the benefit at 10% of the computational cost.

4. **Why Confidence Reweighting?**: Initial experiments without reweighting led to models that predicted zero objects for all scenes - the "lazy" solution to minimize false positives.

Carefully designed loss function is essential for achieving stable training on limited dataset while maintaining detection accuracy across varying object scales and orientations.

## Approach 2: Pre-trained Model Fine-tuning

As an alternative to training a custom model from scratch, I explored fine-tuning the state-of-the-art UniDet3D framework ([https://github.com/filapro/unidet3d](https://github.com/filapro/unidet3d)), which leverages the Superpoint Transformer ([https://github.com/drprojects/superpoint_transformer](https://github.com/drprojects/superpoint_transformer)) for preprocessing. This approach promised faster convergence and better generalization due to pre-trained weights, which are particularly beneficial for small datasets like ours (200 samples). However, it required significant preprocessing overhead and dependency management, as the Superpoint Transformer involves complex data transformations to align RGB and point cloud data.

I successfully set up the UniDet3D framework, including configuring the preprocessing pipeline and integrating it with my dataset structure. However, due to time constraints, I was unable to fully fine-tune the model on my specific dataset. The setup included adapting the data loading to handle our RGB images, point clouds, and bounding box annotations, but further optimization (e.g., hyperparameter tuning, adjusting learning rates) was not completed. This approach remains promising for future work, as the pre-trained weights could potentially outperform the custom model with proper fine-tuning.
