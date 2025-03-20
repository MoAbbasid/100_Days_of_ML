# Logsumexp Trick & Autograd Mathematics

## Logsumexp Optimization (Trick 2)

### Key Implementation
```math
L_i = m_i^{(T_c)} + \log(\ell_i^{(T_c)})
```
Where:
- `m_i` = running maximum per block
- `ℓ_i` = accumulated exponential sums
- Combines two statistics into single log-space value

**Advantages**:
1. Avoids storing separate max/exp-sum values
2. Maintains numerical stability
3. Reduces memory footprint by 50% for statistics

## Autograd Fundamentals

### Derivative Types
| Concept        | Input          | Output         | Mathematical Form |
|----------------|----------------|----------------|--------------------|
| Derivative     | Scalar →       | Scalar →       | $\frac{df}{dx}$    |
| Gradient       | Vector →       | Scalar →       | $\nabla f$         |
| Jacobian       | Vector →       | Vector →       | $J_{ij} = \frac{\partial f_i}{\partial x_j}$ |
| Generalized    | Matrix →       | Matrix →       | Tensor contraction |

### Jacobian Challenges
```math
\text{For } f: \mathbb{R}^N \rightarrow \mathbb{R}^M \quad J \in \mathbb{R}^{M×N}
```
- Memory complexity: $O(MN)$
- Example: For N=1000, M=1000 → 1M elements
- **Sparsity Utilization**:
  - Most deep learning Jacobians are sparse
  - Use matrix-free operations

### Efficient Gradient Computation

#### Chain Rule in Matrix Form
```math
\frac{\partial \phi}{\partial x} = \frac{\partial \phi}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \phi}{\partial y} \cdot W^T \quad [1×N] = [1×M] \cdot [M×N]
```

#### Parameter Gradient
```math
\frac{\partial \phi}{\partial W} = x^T \cdot \frac{\partial \phi}{\partial y} \quad [D×M] = [D×1] \cdot [1×M]
```
Where:
- $x \in \mathbb{R}^D$ (input)
- $W \in \mathbb{R}^{D×M}$ (weight matrix)
- $y = xW \in \mathbb{R}^M$

### Memory Optimization Tricks

1. **Transpose Trick**:
   ```python
   # Instead of storing full Jacobian J ∈ ℝ^{M×N}
   grad = input.T @ output_grad  # ℝ^{D×M} = ℝ^{D×N} @ ℝ^{N×M}
   ```

2. **Shape Preservation**:
   ```python
   # Gradient shape matches parameter shape
   weight_grad = x.t() @ delta  # ℝ^{D×M} = ℝ^{D×N} @ ℝ^{N×M}
   ```

3. **Implicit Jacobian**:
   ```math
fill later   ```
   - B = batch size
   - Avoids materializing $J ∈ ℝ^{B×M×D×M}$

### PyTorch Autograd Specifics

```python
class EfficientMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(x, W)
        return x @ W
    
    @staticmethod
    def backward(ctx, grad_output):
        x, W = ctx.saved_tensors
        grad_x = grad_output @ W.T  # ℝ^{B×D} = ℝ^{B×M} @ ℝ^{M×D}
        grad_W = x.T @ grad_output  # ℝ^{D×M} = ℝ^{D×B} @ ℝ^{B×M}
        return grad_x, grad_W
```

**Key Features**:
- Automatic shape matching
- Sparse Jacobian detection
- Memory-efficient gradient accumulation

## Implementation Considerations

### Dimension Matching Table
| Operation         | Input Dims | Weight Dims | Output Dims |
|-------------------|------------|-------------|-------------|
| Forward Pass      | [B×D]      | [D×M]       | [B×M]       |
| Backward Pass     | [B×M]      | [D×M]       | [B×D]       |

### Memory Complexity
| Approach          | Memory          | Computation       |
|-------------------|-----------------|-------------------|
| Naive Jacobian    | O(MND)          | O(MND)            |
| Efficient Autograd| O(MD + BD + BM) | O(BMD)            |

**Example**: For B=64, M=1024, D=768
- Naive: ~503MB (64*1024*768*4 bytes)
- Efficient: ~12MB ((1024*768 + 64*768 + 64*1024)*4 bytes)

## Summary of Key Insights

1. **Logsumexp Trick**:
   - Combines max and sum statistics in log-space
   - Critical for attention mechanism stability

2. **Autograd Optimization**:
   - Never materialize full Jacobians
   - Leverage matrix transpose properties
   - Maintain gradient shape congruence

3. **Memory Hierarchy**:
   ```text
   Registers > Shared Memory > Global Memory
   (Jacobian Components)    (Full Jacobian)
   ```

4. **Practical Implementation**:
   - Use batch dimensions for implicit broadcasting
   - Prefer matrix multiplications over element-wise ops
   - Exploit weight transpose in backward passes
```
