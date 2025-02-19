# Vision Transformer (ViT)
![image](https://github.com/user-attachments/assets/0f9db44a-94f5-4ca4-a578-ce90f448355f)

A deep dive into Vision Transformers: understanding how transformer architectures can be adapted for computer vision tasks.

## Overview

Vision Transformer (ViT) adapts the transformer architecture, originally designed for sequential natural language processing, to handle image data. Unlike Convolutional Neural Networks (CNNs), ViTs process images by splitting them into fixed-size patches and treating them as a sequence of tokens, similar to how transformers process words in text.

## Core Architecture

### Image Patching and Embedding
<img width="1028" alt="image" src="https://github.com/user-attachments/assets/e3df65f1-5aa6-4d9e-9e53-ed54d4dcbc5f" />

1. **Patch Extraction**: 
   - Images are divided into fixed-size patches
   - These patches form the sequence input for the transformer
   - Each patch is flattened and projected to the embedding dimension using a fully connected neural network

2. **Token Management**:
   - A special [CLS] token is prepended to the sequence
   - Initially initialized as a dummy token
   - Through training, becomes a learned global feature extractor
   - Used for final classification/prediction

3. **Positional Embedding**:
   - Added (not concatenated) to patch embeddings
   - Learned during training due to fixed patch size
   - Crucial because attention is position-independent
   - Implemented as learnable parameters: `[1, num_patches + 1, embedding_dim]`

### Transformer Encoder

#### Multi-Head Self-Attention (MHSA)
- Scaled dot-product attention mechanism
- Input transformed into Query, Key, and Value matrices
- Parallel attention heads capture different types of relationships
- Linear transformation applied post-attention

#### Architecture Details
- Layer Normalization applied before the main functions (pre-norm design)
- Feed-Forward Network (FFN):
  - Linear → GELU → Dropout → Linear → Dropout
- Residual (skip) connections throughout the network:
  - Improves information flow
  - Helps prevent vanishing gradients

## Comparison with CNNs

### Key Differences
- **Inductive Bias**:
  - CNNs: Strong inductive bias towards translation invariance
  - ViTs: Minimal built-in inductive biases
  
- **Information Processing**:
  - CNNs: Hierarchical processing (pyramid structure)
  - ViTs: Global processing from the start

- **Efficiency**:
  - CNNs: Generally more computationally efficient
  - ViTs: Require more data and compute resources

### Advantages
- More flexible at capturing global relationships
- Built-in interpretability through attention mechanisms
- Can handle varying types of visual relationships

## Advanced Topics

### Shifted Window Transformer (Swin)
![image](https://github.com/user-attachments/assets/55392c2d-b818-4f96-a37b-8457ee4149dc)

- Hierarchical feature learning (similar to CNN pyramid)
- Attention computed within local windows
- Windows shift between consecutive layers
- Improves efficiency while maintaining global context

### Data-Efficient Training
- Teacher-student distillation approach
- ConvNet teacher provides inductive biases
- [CLS] token trained to predict teacher model's labels
- Reduces data requirements while maintaining performance

## References
- Original ViT Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Swin Transformer: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- DeiT: "Training data-efficient image transformers & distillation through attention"

---
*Implementation details and code will be added in future updates.*
