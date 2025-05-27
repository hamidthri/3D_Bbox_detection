# 3D Object Detection on RGB-D Data: Technical Documentation

## Project Overview

This project tackles 3D object detection using multimodal RGB-D data consisting of 200 samples. Given the limited dataset size, I explored two distinct approaches: leveraging pre-trained models through fine-tuning and training from scratch with custom architectures.

### Approach 1: Pre-trained Model Fine-tuning

The first approach utilized the state-of-the-art UniDet3D framework ([https://github.com/filapro/unidet3d](https://github.com/filapro/unidet3d)), which requires preprocessing through the Superpoint Transformer ([https://github.com/drprojects/superpoint_transformer](https://github.com/drprojects/superpoint_transformer)). This approach promised faster convergence and better generalization due to pre-trained weights, but required significant data preprocessing overhead and dependency management.

### Approach 2: Custom Training from Scratch

The second approach involved developing a custom multimodal architecture from the ground up. While this provided greater control over the model design and training process, it presented significant challenges given the small dataset size (200 samples), making it particularly difficult to achieve robust generalization without overfitting.

## Data Preprocessing

### Augmentation Constraints

A critical challenge emerged during preprocessing: without access to camera intrinsics and texture information, traditional 3D augmentation techniques (rotation, translation) were not feasible as they would invalidate the correspondence between RGB images and 3D bounding boxes. This constraint significantly limited our augmentation strategy.

**Solution**: I implemented conservative augmentation techniques that preserve spatial relationships:
- Color jittering (brightness, contrast, saturation)
- Random grayscale conversion
- Gaussian blur
- Additive Gaussian noise

### Bounding Box Representation Conversion

The original dataset provided 3D bounding boxes as 8-corner coordinates. To facilitate neural network training and reduce prediction complexity, I converted these to parametric representation: (center, size, quaternion).

**Benefits of this conversion**:
- Reduced prediction dimensionality (10 parameters vs 24 coordinates)
- More intuitive for regression losses
- Eliminates need for post-processing to determine box properties

**Implementation challenges**: Developing a robust corner-to-parametric conversion algorithm that handles edge cases and maintains numerical stability across varying box orientations and sizes.

## Architecture Exploration

I systematically evaluated several state-of-the-art point cloud processing architectures:

### Point Cloud Backbones Evaluated

- **PointNet++**: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" - [Original Paper](https://arxiv.org/abs/1706.02413) | [Primary Repository](https://github.com/charlesq34/pointnet2)

- **DGCNN**: "Dynamic Graph CNN for Learning on Point Clouds" - [Original Paper](https://arxiv.org/abs/1801.07829) | [Primary Repository](https://github.com/WangYueFt/dgcnn)

- **Sparse Convolutional Networks (Minkowski Engine)**: "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks" - [Original Paper](https://arxiv.org/abs/1904.08755) | [Primary Repository](https://github.com/NVIDIA/MinkowskiEngine)

- **PointPillars**: "PointPillars: Fast Encoders for Object Detection from Point Clouds" - [Original Paper](https://arxiv.org/abs/1812.05784) | [Primary Repository](https://github.com/nutonomy/second.pytorch)

After thorough evaluation considering computational efficiency, feature quality, and compatibility with our limited dataset, **DGCNN** was selected as the optimal backbone for point cloud processing due to its dynamic graph construction capability and robust performance on small datasets.

## Final Architecture

The implemented solution follows a multimodal fusion approach:

### RGB Processing
- **Backbone**: Pre-trained EfficientNet-B3 (frozen weights)
- **Feature extraction**: 256-dimensional representations
- **Augmentation**: Color-space transformations preserving spatial integrity

### Point Cloud Processing
- **Architecture**: DGCNN-based feature extractor
- **K-NN graph construction**: k=20 neighbors
- **Output**: 256-dimensional global features

### Multimodal Fusion
- **Method**: Transformer-based attention mechanism
- **Layers**: 4 transformer encoder layers
- **Integration**: Late fusion of RGB and point cloud features

### Prediction Heads
- **Bounding Box Head**: Predicts (center, size, quaternion) for up to 21 objects
- **Confidence Head**: Per-object detection confidence scores

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