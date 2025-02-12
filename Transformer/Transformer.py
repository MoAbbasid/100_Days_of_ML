import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    Converts input tokens (words) into continuous vector representations.
    
    Args:
        d_model (int): The dimensionality of the embedding vectors
        vocab_size (int): Size of the vocabulary
        
    Input Shape:
        x: (batch_size, seq_length) - Integer tensor of token indices
        
    Output Shape:
        (batch_size, seq_length, d_model) - Float tensor of embedded vectors
        
    Notes:
        - Multiplies the embeddings by sqrt(d_model) to scale the embeddings as per the paper
        "Attention Is All You Need" to prevent the dot products from growing too large
    """
    def __init__(self, d_model:int, vocab_size:int):
         super().__init__()
         self.embedding = nn.Embedding(vocab_size, d_model)
         self.d_model = d_model

    def forward(self, x):
        """
        Convert token indices to embeddings and scale them.
        
        Args:
            x (torch.Tensor): Input tensor of token indices, shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Scaled embeddings, shape (batch_size, seq_length, d_model)
        """
        return self.embedding(x.long()) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings since the transformer has no inherent
    understanding of token order.
    
    Args:
        d_model (int): Dimensionality of the model
        seq_len (int): Maximum sequence length
        dropout (float): Dropout rate for regularization
        
    Input Shape:
        x: (batch_size, seq_length, d_model) - Embedded input sequences
        
    Output Shape:
        (batch_size, seq_length, d_model) - Embedded sequences with positional encoding
        
    Notes:
        - Uses sine and cosine functions of different frequencies to encode position
        - Even indices use sine, odd indices use cosine
        - The positional encodings have the same dimension as the embeddings so they can be summed
        - The wavelengths form a geometric progression from 2π to 10000·2π
    """
    def __init__(self, d_model: int,seq_len: int, dropout: float) -> None:
         super().__init__()  
         self.d_model = d_model
         self.seq_len = seq_len
         # dropout is for preventing overfitting
         self.dropout = nn.Dropout(dropout)
         
         # create positional encoding tensor of shape (seq_len, d_model)
         pe = torch.zeros(self.seq_len, self.d_model)

         # numerator and denominator for pe
         position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0)/ self.d_model))

         # apply sin to even and cos to odd
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)

         # add batch_dim infront
         pe = pe.unsqueeze(0)

         # register buffer to make it persistent and not learnable
         self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings, shape (batch_size, seq_length, d_model)
            
        Returns:
            torch.Tensor: Embeddings with positional encoding, shape (batch_size, seq_length, d_model)
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Normalizes the input tensor for stable training, applying an affine transformation
    with learnable parameters.
    
    Args:
        eps (float): Small constant for numerical stability (default: 1e-6)
        
    Input Shape:
        x: (..., d_model) - Input tensor
        
    Output Shape:
        (..., d_model) - Normalized tensor
        
    Notes:
        - Computes mean and standard deviation across the last dimension
        - Uses learnable parameters alpha (scale) and bias (shift)
        - Normalizes each input to have mean 0 and variance 1 before scaling
    """
    def __init__(self, eps:float = 10**-6) -> None:
         super().__init__()
         self.eps = eps
         self.alpha = nn.Parameter(torch.ones(1))
         self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Apply layer normalization to input.
        
        Args:
            x (torch.Tensor): Input tensor to normalize
            
        Returns:
            torch.Tensor: Normalized tensor with same shape as input
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Applies two linear transformations with a ReLU activation in between.
    Part of each encoder layer that processes the attention output.
    
    Args:
        d_model (int): Input and output dimensionality
        dff (int): Hidden layer dimensionality (usually 4*d_model)
        dropout (float): Dropout rate for regularization
        
    Input Shape:
        x: (batch_size, seq_length, d_model)
        
    Output Shape:
        (batch_size, seq_length, d_model)
        
    Notes:
        - First linear layer expands the dimension to dff
        - ReLU activation removes negative values
        - Second linear layer projects back to d_model dimensions
        - Includes dropout for regularization
    """
    def __init__(self, d_model:int, dff:int, dropout:float) -> None:
         super().__init__()
         self.linear1 = nn.Linear(d_model, dff)
         self.dropout = nn.Dropout(dropout)
         self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        """
        Apply feed-forward transformation to input.
        
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_length, d_model)
            
        Returns:
            torch.Tensor: Transformed tensor, shape (batch_size, seq_length, d_model)
        """
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    """
    Allows the model to jointly attend to information from different representation subspaces.
    
    Args:
        d_model (int): Model's dimension
        h (int): Number of attention heads
        dropout (float): Dropout rate
        
    Input Shape:
        q, k, v: (batch_size, seq_length, d_model) - Query, Key, and Value matrices
        mask: (batch_size, 1, seq_length, seq_length) or None - Optional attention mask
        
    Output Shape:
        (batch_size, seq_length, d_model) - Attended feature representations
        
    Notes:
        - Splits d_model into h heads, each with dimension d_k = d_model/h
        - Each head performs scaled dot-product attention independently
        - Heads are concatenated and linearly transformed to produce final output
        - Uses separate linear layers for Q, K, V projections
        - Attention scores are scaled by 1/sqrt(d_k) to prevent softmax saturation
    """
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
         super().__init__()
         self.d_model = d_model
         self.h = h
         self.dropout = nn.Dropout(dropout) 

         assert d_model % h == 0, "d_model is not divisible by h"

         self.d_k = d_model // h
         self.w_q = nn.Linear(d_model, d_model) # Wq
         self.w_k = nn.Linear(d_model, d_model) # Wk
         self.w_v = nn.Linear(d_model, d_model) # Wv
         self.w_0 = nn.Linear(d_model, d_model) # W0 

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.
        
        Args:
            query (torch.Tensor): Query tensor, shape (batch, h, seq_len, d_k)
            key (torch.Tensor): Key tensor, shape (batch, h, seq_len, d_k)
            value (torch.Tensor): Value tensor, shape (batch, h, seq_len, d_k)
            mask (torch.Tensor): Optional mask tensor
            dropout (nn.Dropout): Dropout module
            
        Returns:
            tuple: (attended values, attention scores)
        """
        d_k = query.shape[-1]

        # (batch, h, Seq_len, d_k) --> (batch, h,  Seq_len, Seq_len)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
             attention_score = attention_score.masked_fill(mask ==0, -1e9)
        attention_score = attention_score.softmax(dim=-1) 
        if dropout is not None:
             attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        """
        Compute multi-head attention.
        
        Args:
            q (torch.Tensor): Query tensor, shape (batch_size, seq_length, d_model)
            k (torch.Tensor): Key tensor, shape (batch_size, seq_length, d_model)
            v (torch.Tensor): Value tensor, shape (batch_size, seq_length, d_model)
            mask (torch.Tensor): Optional attention mask
            
        Returns:
            torch.Tensor: Attended tensor, shape (batch_size, seq_length, d_model)
        """
        query = self.w_q(q) # (batch, Seq_len, d_model) --> (batch, Seq_len, d_model)
        key = self.w_k(k) # (batch, Seq_len, d_model) --> (batch, Seq_len, d_model)
        value = self.w_v(v) # (batch, Seq_len, d_model) --> (batch, Seq_len, d_model)

        # divide (batch, Seq_len, d_model) --> (batch, Seq_len, h, dk) then transpose to (batch, h, Seq_len, dk)
        # to pass h to each head so it can see (Seq_len, dk) which is all of the sentence but different embeddings
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # apply self attention
        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, Seq_len, dk) --> (batch, Seq_len, h, dk) --> (batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)
         
         # Multiply by output matrix Wo (batch, Seq_len, d_model) --> (batch, Seq_len, d_model)
        return self.w_0(x)

class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization.
    Helps with training deep networks by allowing gradients to flow directly through skip connections.
    
    Args:
        dropout (float): Dropout rate
        
    Input Shape:
        x: (batch_size, seq_length, d_model) - Input tensor
        sublayer: Callable - The layer to apply (attention or feed-forward)
    """
    def __init__(self, dropout:float) -> None:
         super().__init__()  
         self.dropout = nn.Dropout(dropout)
         self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
    Output Shape:
        (batch_size, seq_length, d_model) - Output with residual connection
        
    Notes:
        - Applies layer normalization before the sublayer (pre-norm formulation)
        - Adds the sublayer output to the original input (residual connection)
        - Dropout is applied to the sublayer output before addition
        """
        return x + self.dropout(sublayer(self.layer_norm(x)))

class EncoderBlock(nn.Module):
    """
    A single encoder block combining self-attention and feed-forward layers with residual connections.
    
    Args:
        self_attention_block (MultiHeadAttention): Self-attention mechanism
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward network
        dropout (float): Dropout rate
        
    Input Shape:
        x: (batch_size, seq_length, d_model) - Input sequences
        src_mask: (batch_size, 1, seq_length, seq_length) - Source padding mask
    """
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
         super().__init__()  
         self.self_attention = self_attention_block
         self.feed_forward = feed_forward_block
         self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
    Output Shape:
        (batch_size, seq_length, d_model) - Processed sequences
        
    Notes:
        - First applies multi-head self-attention with residual connection
        - Then applies feed-forward network with another residual connection
        - Each residual connection includes dropout and layer normalization
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward(x))
        return x

class Encoder(nn.Module):
    """
    Complete encoder consisting of a stack of identical encoder blocks.
    
    Args:
        layers (nn.ModuleList): List of EncoderBlock instances
        
    Input Shape:
        x: (batch_size, seq_length, d_model) - Input sequences
        mask: (batch_size, 1, seq_length, seq_length) - Attention mask
    
    """
    def __init__(self, layers:nn.ModuleList) -> None:
         super().__init__()
         self.layers = layers
         self.layer_norm = LayerNormalization()

    def forward(self, x, mask):
        """
     Output Shape:
        (batch_size, seq_length, d_model) - Final encoded representations
        
    Notes:
        - Processes input through multiple encoder blocks sequentially
        - Each block maintains the same dimensionality
        - Final layer normalization is applied to the output
        - The mask prevents attention to padding tokens and/or future tokens
        """
        for layer in self.layers:
             x = layer(x, mask)
        return self.layer_norm(x)  
    
class DecoderBlock(nn.Module):
    """
    A single decoder block combining self-attention, cross-attention, and feed-forward layers.
    
    Args:
        self_attention (MultiHeadAttention): Self-attention mechanism for target sequence
        cross_attention (MultiHeadAttention): Cross-attention mechanism to attend to encoder outputs
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward network
        dropout (float): Dropout rate
        
    Input Shape:
        x: (batch_size, seq_length, d_model) - Target sequence
        encoder_output: (batch_size, src_seq_length, d_model) - Encoder's output
        src_mask: (batch_size, 1, src_seq_length) - Source padding mask
        tgt_mask: (batch_size, 1, seq_length) - Target sequence mask (prevents attending to future tokens)
        
    Output Shape:
        (batch_size, seq_length, d_model) - Processed target sequence
        
    Notes:
        - Contains three sub-layers: self-attention, cross-attention, and feed-forward
        - Each sub-layer has its own residual connection and layer normalization
        - Self-attention allows target sequence to attend to its own previous tokens
        - Cross-attention allows target sequence to attend to all encoder outputs
    """
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, 
                 feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward(x))
        return x

class Decoder(nn.Module):
    """
    Complete decoder consisting of a stack of identical decoder blocks.
    
    Args:
        layers (nn.ModuleList): List of DecoderBlock instances
        
    Input Shape:
        x: (batch_size, seq_length, d_model) - Target sequence
        encoder_output: (batch_size, src_seq_length, d_model) - Encoder's output
        src_mask: (batch_size, 1, src_seq_length) - Source padding mask
        tgt_mask: (batch_size, 1, seq_length) - Target sequence mask
        
    Output Shape:
        (batch_size, seq_length, d_model) - Final decoded representations
        
    Notes:
        - Processes input through multiple decoder blocks sequentially
        - Each block maintains the same dimensionality
        - Final layer normalization is applied to the output
        - Uses masking to prevent attending to future tokens during training
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)

class ProjectionLayer(nn.Module):
    """
    Projects decoder output to vocabulary size and applies log softmax.
    
    Args:
        d_model (int): Input dimensionality from decoder
        vocab_size (int): Size of target vocabulary
        
    Input Shape:
        x: (batch_size, seq_length, d_model) - Decoder output
        
    Output Shape:
        (batch_size, seq_length, vocab_size) - Log probabilities over target vocabulary
        
    Notes:
        - Transforms decoder outputs into vocabulary space
        - Applies log softmax for numerical stability
        - Used for final prediction of target tokens
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim= -1)
        
class Transformer(nn.ModuleList):
    """
    Complete Transformer model combining encoder and decoder for sequence-to-sequence tasks.
    
    Args:
        encoder (Encoder): The encoder stack
        decoder (Decoder): The decoder stack
        src_emb (InputEmbedding): Source sequence embedding layer
        tgt_emb (InputEmbedding): Target sequence embedding layer
        src_pos (PositionalEncoding): Positional encoding for source
        tgt_pos (PositionalEncoding): Positional encoding for target
        projection (ProjectionLayer): Output projection layer
        
    Input Shape:
        src: (batch_size, src_seq_length) - Source sequence
        tgt: (batch_size, tgt_seq_length) - Target sequence
        src_mask: (batch_size, 1, src_seq_length) - Source padding mask
        tgt_mask: (batch_size, 1, tgt_seq_length) - Target sequence mask
        
    Output Shape:
        (batch_size, tgt_seq_length, target_vocab_size) - Log probabilities over target vocabulary
        
    Notes:
        - Implements the full Transformer architecture from "Attention Is All You Need"
        - Suitable for tasks like translation, summarization, and text generation
        - Uses separate embedding layers for source and target sequences
        - Applies positional encoding to both source and target embeddings
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: InputEmbedding, tgt_emb: InputEmbedding,
                  src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src, src_mask):
        """
        Encode source sequence.
        
        Args:
            src (torch.Tensor): Source sequence
            src_mask (torch.Tensor): Source padding mask
            
        Returns:
            torch.Tensor: Encoded representation of source sequence
        """
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decode target sequence given encoded source.
        
        Args:
            encoder_output (torch.Tensor): Encoded source sequence
            src_mask (torch.Tensor): Source padding mask
            tgt (torch.Tensor): Target sequence
            tgt_mask (torch.Tensor): Target sequence mask
            
        Returns:
            torch.Tensor: Decoded representation of target sequence
        """
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """
        Project decoder output to vocabulary space.
        
        Args:
            x (torch.Tensor): Decoder output
            
        Returns:
            torch.Tensor: Log probabilities over target vocabulary
        """
        return self.projection(x)

  
def build_fn(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, 
             d_model:int = 512, Nblocks: int = 6, heads: int = 8, dropout:float = 0.1, dff:int = 2048) -> Transformer:
    """
    Build a complete Transformer model with specified parameters.
    
    Args:
        src_vocab_size (int): Size of source vocabulary
        tgt_vocab_size (int): Size of target vocabulary
        src_seq_len (int): Maximum source sequence length
        tgt_seq_len (int): Maximum target sequence length
        d_model (int, optional): Model dimension. Defaults to 512
        Nblocks (int, optional): Number of encoder/decoder blocks. Defaults to 6
        heads (int, optional): Number of attention heads. Defaults to 8
        dropout (float, optional): Dropout rate. Defaults to 0.1
        dff (int, optional): Feed-forward hidden dimension. Defaults to 2048
        
    Returns:
        Transformer: Initialized transformer model
        
    Notes:
        - Creates embedding layers for source and target sequences
        - Builds encoder and decoder stacks with specified number of blocks
        - Initializes parameters using Xavier uniform initialization
        - Uses default hyperparameters from the original Transformer paper
    """
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(Nblocks):
        encoder_self_attention = MultiHeadAttention(d_model, heads, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, encoder_feed_forward, dropout)

        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(Nblocks):
        decoder_self_attention = MultiHeadAttention(d_model, heads, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, heads, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout)

        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer