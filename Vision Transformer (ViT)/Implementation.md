# Vision Transformer (ViT) Implementation & Architecture

## Essential Imports

- **Required Libraries:**  
  - `einops` (for rearranging tensors)
  - `tqdm` (for progress bars)
  - `torch` and `torchvision` (for model building and dataset handling)
  - `torchvision.transforms` (for data preprocessing)
  - `torchsummary` (optional, for model summary visualization)

## 1. ViT Architecture Overview

**Input Embedding (Patchify > Linear Projection)**
- **Purpose:** Convert an image into a sequence of patch embeddings.
- **Key Steps:**
  - **Patchify:** Use `einops.rearrange` to split the image into patches.
  - **Linear Projection:** Flatten each patch (input size = P × P × C) and project it to a fixed latent dimension.
  - **CLS Token Prepending:** Concatenate the `cls_token` to the beginning of the patch embeddings.
  - **Positional Embedding Addition:** Use `einops.repeat` to expand `pos_emb` to match the number of tokens and add it to the tokens.
  - **Optional:** Call `torch.cuda.empty_cache()` if needed for GPU memory management.

---

- **Transformer Encoder Block:**
  - **Multi-Head Self-Attention:**
    - Compute **Q, K, V** via separate linear layers.
    - Reshape the resulting tensors to separate multiple heads (dimensions: `batch, heads, tokens, head_dim`).
    - Calculate attention scores with scaled dot-product attention and apply softmax.
    - Multiply the attention weights by **V** and then merge the heads back into a single tensor.
  - **Residual Connections & LayerNorm:**
    - Apply residual (skip) connections around both the self-attention sub-layer and the feed-forward network (FFN).
    - Use separate `LayerNorm` layers after the attention and FFN components.
  - **Feed-Forward Network (FFN):**
    - Expand the embedding size (e.g., to 4× latent size), apply an activation function (GELU), and dropout.
    - Project the expanded features back to the original latent size.

- **Model Variants:**
  - **Custom Encoder Blocks:** 
    - Implement and stack your own encoder blocks using an `nn.ModuleList`—ideal for learning the internal mechanics.
  - **Built-In PyTorch Modules:**
    - Alternatively, use `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` for a more concise, optimized implementation.
  - **Final Classification Head:**
    - Extract the CLS token (usually the first token) after the encoder stack.
    - Pass the CLS token through a small MLP (typically LayerNorm followed by a Linear layer) to obtain class scores.

## 2. Data Preprocessing & Transforms

- **Transforms Pipeline:**
  - Use `transforms.Compose` to chain together a series of transformations (e.g., `Resize`, `ToTensor`, `Normalize`).
  - **Resize:** Adjust images to a uniform size for batch processing.
  - **ToTensor:** Convert images to PyTorch tensors, scaling pixel values to the range [0, 1].
  - **Normalize:** Standardize images by subtracting the mean and dividing by the standard deviation per channel.  
    - For example, `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` scales images to roughly the range [-1, 1].

- **Custom vs. Standard Datasets:**
  - **Standard Datasets:** Can be used directly with PyTorch's DataLoader (e.g., CIFAR-10/100, MNIST).
  - **Custom Dataset Classes:** Create custom dataset classes when your data is in a non-standard format or when specific preprocessing is required.

## 3. DataLoader Settings

- **`num_workers`:**
  - Optional parameter that determines how many subprocesses are used for data loading.
  - Default is `0` (data loading in the main process). Increasing this can speed up data loading depending on your system.
  - Optimal value depends on dataset size, storage speed, and CPU cores.

- **Batch-Size Agnostic Parameters:**
  - Define learnable tokens (e.g., CLS token and positional embeddings) with a batch dimension of `1`.
  - Expand these tokens at runtime to match the actual batch size, ensuring flexibility.

## 4. Debugging Tips

- **Print Tensor Shapes:**
  - At each stage (e.g., after patch extraction, linear projection, attention), print out tensor shapes to verify they are as expected.
- **Patch Size Consistency:**
  - Ensure that the chosen patch size divides the image dimensions evenly, or adjust using padding.
- **Consistent Naming & Hyperparameters:**
  - Pass all key hyperparameters (patch size, latent size, number of heads, etc.) to your model to avoid reliance on global variables and to improve clarity.
