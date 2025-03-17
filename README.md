# 100 Days of Deep Learning

This repository documents my journey through 100 days of learning and implementing various deep learning models and techniques. The goal is to strengthen my understanding of machine learning algorithms, architectures, and practical implementations with PyTorch.

## Learning Focus Areas

- Deep Learning Architectures (Transformers, CNNs)
- PyTorch Implementation Techniques
- Machine Learning Systems Design
- Model Optimization

## Progress Log

| Day      | Progress & Notes | Status |
|----------|-----------------|--------|
| Day 1    | Started implementing the Transformer model from scratch following "Attention is All You Need" paper. Created core components: multi-head attention mechanism with scaled dot-product attention, positional encoding with sine and cosine functions. Implemented layer normalization and the feed-forward network with ReLU activation. | ✅ Completed |
| Day 2    | Continued Transformer implementation. Built the encoder stack with multi-head attention, residual connections, and layer normalization. Created the decoder with masked multi-head attention and encoder-decoder attention. Designed the final linear and softmax output layers. | ✅ Completed |
| Day 3    | Completed and tested the Transformer model implementation. Added input/output embedding with weight sharing. Created utilities for sequence padding and masking. Verified model output shapes and connections. Examined the complexity and architecture of the full transformer pipeline. | ✅ Completed |
| Day 4    | Started implementing Vision Transformer (ViT) architecture. Created patch embedding layer to convert images into sequence of embeddings. Built positional embedding layer for maintaining spatial information. Added class token for image classification. | ✅ Completed |
| Day 5    | Continued ViT implementation. Constructed transformer encoder with self-attention blocks and MLP layers. Added layer normalization layers at appropriate positions. Implemented residual connections throughout the architecture. | ✅ Completed |
| Day 6    | Completed ViT implementation and conducted basic testing. Verified patch embedding functionality, positional encoding, and transformer blocks. Added classification head (MLP) with layer normalization. Ensured proper dimensions throughout the network. Documented architecture and implementation details. | ✅ Completed |
| Day 7    | Beginning Flash Attention implementation to optimize transformer processing speed and memory usage. | 🔄 In Progress |
| Day 8    |  |  |

## Current Focus: Flash Attention Implementation

I'm currently working on implementing Flash Attention to improve the efficiency of transformer models. This involves:
- Understanding the mathematical foundations of the algorithm
- Creating an efficient PyTorch implementation
- Benchmarking against standard attention mechanisms
- Applying the implementation to existing transformer models

## Next Topics in Queue

1. Mixture of Experts (MoE) implementation
2. Mamba (State Space Models)
3. ML system design principles
4. Model quantization and optimization techniques
5. End-to-end object detection project with custom architecture

## Resources

- Papers:
  - "Attention Is All You Need" (Original Transformer paper)
  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT paper)
  - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

- Books:
  - "Designing Machine Learning Systems" by Chip Huyen
  - "Deep Learning" by Goodfellow, Bengio, and Courville

## Goals

- Implement 15 different deep learning models from scratch
- Create 5 end-to-end projects with practical applications
- Write 10 blog posts about my implementations and learnings
- Build a comprehensive understanding of modern ML architectures
- Develop skills in ML systems design and optimization
