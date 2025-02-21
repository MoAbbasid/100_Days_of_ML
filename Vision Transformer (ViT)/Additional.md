# Advanced Concepts: Positional Encodings, Orthogonality, & PyTorch Notes
- ** Randomly drawn vector are in higher dimensions are always nearly orthogonal**
## 1. Orthogonality Concepts

- **Geometric:** Vectors are orthogonal if they are perpendicular.
- **Algebraic:** Two vectors are orthogonal if their dot product equals zero.
- **Statistical:** Orthogonal (or independent) variables do not interfere with each other.
- **Signal Processing:** Signals are orthogonal if they do not interfere, allowing them to be separated.
- **General Idea:** In our context, “orthogonal” often means the positional encodings and token embeddings are (approximately) independent or non-interfering.

## 2. Positional Encodings & Their Addition

- **Intuition:**  
  - Although it might seem counterintuitive to simply add positional encodings to token embeddings, in high-dimensional spaces, this addition allows the model to combine content and positional information effectively.
  - The addition leverages the properties of high-dimensional spaces, where different types of information (content vs. position) can reside in approximately orthogonal subspaces.

- **Mathematical Insight:**  
  - Consider the expression:  
    ```
    (Q(x+e))' (K(y+f)) = (Qx + Qe)' (Ky + Kf)
                       = (Qx)'Ky + (Qx)'Kf + (Qe)'Ky + (Qe)'Kf
    ```
  - **Interpretation:**  
    - The term `(Qx)'Ky` reflects the traditional attention score between tokens.
    - The additional terms account for interactions between token content and positional information:
      - `(Qx)'Kf`: How much attention should we pay to token x given the position of y.
      - `(Qe)'Ky`: How much attention should we pay to token y given the position of x.
      - `(Qe)'Kf`: How the positional information of both tokens interacts.

- **Dimensionality & Orthogonality:**  
  - Increasing the embedding dimension generally increases the chance that different information (content and position) lies in approximately orthogonal subspaces.
  - If word embeddings and positional encodings lie in separate, smaller-dimensional subspaces, these subspaces might be approximately orthogonal, allowing the model to combine them efficiently.
 
- **Refs**
- https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/
- https://youtu.be/k4CxJLXc3-0
