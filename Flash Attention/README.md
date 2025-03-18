# Flash Attention Notes

## Key Concepts

- Built on top of **Multi-Head Attention** mechanism
- Focuses on optimizing memory access patterns rather than matrix multiplications
  - Matrix multiplications (MatMul) are already well-optimized
  - Doesn't optimize projection operations at all

## I/O Bound Nature of Attention

- Primary bottleneck is memory access (I/O bound)
- GPU global memory (HBM) is slow → solution uses shared memory (SRAM)
- Utilizes **tiling** approach similar to PMPP book principles
  - Blocked computation helps reduce memory accesses

## Core Algorithm (FlashAttention-2)

![Pasted Graphic copy](https://github.com/user-attachments/assets/4ccf5aa6-b159-4f5d-94a7-fdab7485d8d3)

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

## Online Softmax Mechanics

![image](https://github.com/user-attachments/assets/86cfc893-c767-4a87-bbde-d05850c32602)

**Key Equations for Block Updates**:

1. Maximum Update:  

2. Exponential Correction:  

3. Log-Sum-Exp Update:  

4. Output Update:  

## Numerical Stability

- Maintains numerical stability through:
  1. **Running Maximum** (m_i^(j)) 
  2. **Exponential Correction Terms**
  3. **Log-Sum-Exp Tracking** (ℓ_i^(j))


## Memory Hierarchy Optimization

| Operation           | Location  | Access Cost |
|---------------------|-----------|-------------|
| Matrix Multiplications | SRAM     | Fast        |
| Statistics Tracking | Registers | Fastest     |
| Block Transfers     | HBM ↔ SRAM| Slow        |

## Performance Characteristics

- **Memory Complexity**: O(N) vs O(N²) for standard attention
- **Speed**: 2-4× faster than baseline implementations
- **Perfect for**:
  - Long sequences (≥4K tokens)
  - Memory-constrained environments
  - Training scenarios with large batch sizes
