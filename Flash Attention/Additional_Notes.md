
# Flash Attention Technical Documentation

## Core Algorithm Implementation
![image](https://github.com/user-attachments/assets/bece632e-1cd6-4eb2-bf35-db462e49768b)

### FlashAttention-2 Forward Pass
```python
Algorithm 1: FlashAttention-2 Forward Pass

Input: Q, K, V ∈ ℝ^(N×d) in HBM, block sizes B_c, B_r
Output: O ∈ ℝ^(N×d), L ∈ ℝ^N

1. Partition Q into T_r = ⌈N/B_r⌉ blocks Q_1,...,Q_T
2. Partition K,V into T_c = ⌈N/B_c⌉ blocks K_1,...,K_Tc and V_1,...,V_Tc
3. Initialize output blocks O_i and logsumexp blocks L_i

for each Q_i block (1 ≤ i ≤ T_r):
    Load Q_i from HBM → SRAM
    Initialize:
        O_i^(0) = 0 ∈ ℝ^(B_r×d)
        ℓ_i^(0) = 0 ∈ ℝ^B_r
        m_i^(0) = -∞ ∈ ℝ^B_r
    
    for each K_j,V_j block (1 ≤ j ≤ T_c):
        Load K_j,V_j from HBM → SRAM
        Compute S_ij = Q_iK_j^T ∈ ℝ^(B_r×B_c)
        Update statistics:
            m_i^(j) = max(m_i^(j-1), rowmax(S_ij))
            P̃_ij = exp(S_ij - m_i^(j))
            ℓ_i^(j) = e^(m_i^(j-1)-m_i^(j))ℓ_i^(j-1) + rowsum(P̃_ij)
        
        Update output:
            O_i^(j) = diag(e^(m_i^(j-1)-m_i^(j))⁻¹ O_i^(j-1) + P̃_ijV_j
    
    Finalize block:
        O_i = diag(ℓ_i^(Tc))⁻¹ O_i^(Tc)
        L_i = m_i^(Tc) + log(ℓ_i^(Tc))
        Write O_i,L_i to HBM

Return O,L
```

## GPU Architecture Considerations

### CPU vs GPU Paradigms
```text
CPU (Vertical Processing):      GPU (Horizontal Processing):
● Optimized for latency        ● Optimized for throughput
● Sequential execution         ● Massively parallel execution
● Deep cache hierarchy        ● High memory bandwidth
```

### Key GPU Optimizations
- **I/O Bound Solutions**:
  - Block-wise SRAM processing
  - Fused memory operations
- **Boundary Checking**:
  - Prevents warp overflow
  - Essential for SIMD/SIMT execution
- **Parallelization Strategy**:
  ```python
  # Triton parallelization example
  for start_kv in range(lo, hi, BLOCK_SIZE_KV):
      start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
      K_block = tl.load(K_block_ptr)
      QK_block = tl.dot(Q_block, K_block)
  ```

## Tensor Layouts & Memory Management

### Memory Layout Fundamentals
```python
Original Tensor (Row-major):    Transposed Tensor:
Shape: [2,3]                   Shape: [3,2]
Stride: [3,1]                  Stride: [1,3]
```

### Key Memory Concepts
1. **Stride Arithmetic**:
   ```python
   element_address = base_ptr + i*stride[0] + j*stride[1]
   ```
2. **Transpose Optimization**:
   ```python
   # Free transpose via stride manipulation
   K_block_ptr = tl.make_block_ptr(
       shape=(HEAD_DIM, SEQ_LEN),
       strides=(stride_K_dim, stride_K_seq),
       order=(0, 1)
   )
   ```
3. **Non-Contiguous Tensors**:
   - Require memory re-layout for `view()` operations
   - Common in attention score matrices

## CUDA/Triton Implementation Details

### Kernel Configuration
```python
# Block pointer setup for Triton
Q_block_ptr = tl.make_block_ptr(
    base=0 + qwk_offset,
    shape=(SEQ_LEN, HEAD_DIM),
    strides=(stride_Q_seq, stride_Q_dim),
    block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
    order=(1, 0)
```

### Attention Computation Stages
```python
# Stage 1: Compute attention scores
if STAGE == 2:
    mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
    QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1e6)

# Stage 2: Softmax updates
m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
alpha = tl.math.exp(m_i - m_ij)
P_block = tl.math.exp(QK_block - m_ij[:, None])
```

### Memory Access Patterns
```python
# Efficient block loading
V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
K_block_ptr = tl.advance(K_block_ptr, (0, lo))

# Accumulator update
O_block = O_block * alpha[:, None]
O_block = tl.dot(P_block, V_block, acc=O_block)
```

## Performance Optimization Strategies

### Memory Hierarchy
| Memory Type | Access Latency | Bandwidth | Usage |
|-------------|----------------|-----------|-------|
| HBM         | 300-500 cycles | High      | Main storage |
| SRAM        | 20-30 cycles   | Very High | Block processing |
| Registers   | 1 cycle        | Peak      | Local variables |

### Blocking Strategy
```text
Q Blocking:                K/V Blocking:
● B_r = 128-256           ● B_c = 64-128
● T_r = N/B_r             ● T_c = N/B_c
● SRAM Usage: O(B_r*d)    ● SRAM Usage: O(B_c*d)
```

## Numerical Stability Considerations

### Online Softmax Components
1. **Running Maximum**:
   ```math
   m^{(j)} = \max(m^{(j-1)}, \text{rowmax}(S^{(j)}))
   ```
2. **Exponential Correction**:
   ```math
   \tilde{P}^{(j)} = \exp(S^{(j)} - m^{(j)})
   ```
3. **Log-Sum-Exp Tracking**:
   ```math
   \ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \text{rowsum}(\tilde{P}^{(j)})
   ```

## Implementation Notes

1. **Tensor Contiguity**:
   - Transposed tensors require explicit contiguous conversion
   - Use `permute` + `contiguous` for memory layout changes

2. **Mixed Precision**:
   ```python
   P_block = P_block.to(tl.float16)  # Reduce memory bandwidth
   O_block = tl.dot(P_block, V_block, acc=O_block)
   ```

3. **Block Size Selection**:
   - Balance between SRAM usage and parallelism
   - Typical values: BLOCK_SIZE_Q=128, BLOCK_SIZE_KV=64
```
