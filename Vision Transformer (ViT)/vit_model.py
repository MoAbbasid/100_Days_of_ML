import einops
import tqdm
from torchsummary import summary
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

patchsize = 4
latent_size = 768
n_channels = 3
heads = 12
n_encoders = 12
dropout = 0.1
n_classes = 100
img_size = (32, 32)

epochs = 40
base_lr = 10e-4
weight_decay = 0.03
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class InputEmbedding(nn.Module):
    def __init__(self, img_size, patchsize, n_channels, latent_size) -> None:
        super().__init__()
        self.img_size = img_size
        self.patchsize = patchsize
        self.n_channels = n_channels
        self.latent_size = latent_size
        
        # self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Define input size for a single patch
        self.n_patches = (img_size[0] // patchsize) * (img_size[1] // patchsize)
        self.input_size = self.patchsize * self.patchsize * self.n_channels

        self.linear = nn.Linear(self.input_size, self.latent_size)
        
        # Create cls_token for a single item and let batch dimension be handled automatically
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.latent_size))
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches+1, self.latent_size))

    def forward(self, x):
        batch_size = x.shape[0]  # Get batch size from input
        # print("Input shape:", x.shape)  # Should be (batch_size, n_chanbels, h, w)
    
        # Rearrange image into patches
        patches = einops.rearrange(
            x, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', 
            h1=self.patchsize, w1=self.patchsize
        )
        
        # Apply linear transformation
        x_linear = self.linear(patches)
        
        # Expand cls_token to match batch size
        cls_tokens = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        
        # Concatenate cls token with patch embeddings
        x_prepend = torch.cat([cls_tokens, x_linear], dim=1)
        
        # Add positional embeddings
        # pos_embed = einops.repeat(self.pos_emb, '1 1 d -> b m d', 
        #                         b=batch_size, m=x_prepend.shape[1])
        x_prepend += self.pos_emb

        return x_prepend

class EncoderBlock(nn.Module):
    def __init__(self, latent_size, heads, dropout=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.heads = heads
        self.head_dim = latent_size // heads
        
        # Linear layers for Q, K, V and output projection
        self.W_q = nn.Linear(latent_size, latent_size)
        self.W_k = nn.Linear(latent_size, latent_size)
        self.W_v = nn.Linear(latent_size, latent_size)
        self.W_o = nn.Linear(latent_size, latent_size)
        
        # LayerNorms for residual connections
        self.ln1 = nn.LayerNorm(latent_size)
        self.ln2 = nn.LayerNorm(latent_size)
        
        # Feedforward Network (FFN)
        self.mlp_head = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, num_patches, latent_size)
        b, n, d = x.shape
        
        # Multi-Head Self-Attention:
        Q = self.W_q(x)  # (b, n, latent_size)
        K = self.W_k(x)  # (b, n, latent_size)
        V = self.W_v(x)  # (b, n, latent_size)
        
        # Reshape for multi-head attention
        Q = Q.view(b, n, self.heads, self.head_dim).transpose(1, 2)  # (b, heads, n, head_dim)
        K = K.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores and weights
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (b, heads, n, n)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_out = attn_weights @ V  # (b, heads, n, head_dim)
        
        # Merge heads: (b, heads, n, head_dim) -> (b, n, latent_size)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, d)
        attn_out = self.W_o(attn_out)
        
        # Add & Norm: Residual connection for self-attention
        x = self.ln1(x + self.dropout(attn_out))
        
        # Feedforward Network (mlp_head)
        ffn_out = self.mlp_head(x)
        
        # Add & Norm: Residual connection for FFN
        x = self.ln2(x + self.dropout(ffn_out))
        
        return x


class ViT(nn.Module):
  def __init__(self, n_encoders, latent_size, n_classes, dropout=0.01) -> None:
      super().__init__()
      self.n_encoders = n_encoders
      self.latent_size = latent_size
      self.n_classes = n_classes
      self.dropout = dropout

      # Input embedding: patchify + linear projection + positional embedding
      self.input_embedding = InputEmbedding(patchsize, n_channels, latent_size)

      # Stack encoder blocks
      self.encoders = nn.ModuleList([EncoderBlock(latent_size, heads, dropout) for _ in range(n_encoders)])

      # Classification head: use only the CLS token (index 0)
      self.mlp_head = nn.Sequential(
          nn.LayerNorm(normalized_shape=latent_size),
          nn.Linear(in_features=latent_size, out_features=n_classes)
      )

  def forward(self, x):

    # prepare embed
    x = self.input_embedding(x)

    # Encoder Blocks loop
    for encoder in self.encoders:
      x = encoder(x)

    # pass only the token to the classifier
    # print(x.shape)
    x = self.mlp_head(x[:, 0, :])

    return x

# download CIFAR and prepare it

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 pixels
    transforms.ToTensor(),        # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])


# Create DataLoader for the validation dataset

# Download and load the CIFAR-100 training dataset
train_dataset = datasets.CIFAR100(root='./content', train=True, download=True, transform=transform)

# Define the size of the training and validation datasets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Split the train_dataset into train and val datasets
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=62, shuffle=False, num_workers=2, pin_memory=True)

# Download and load the CIFAR-100 test dataset
test_dataset = datasets.CIFAR100(root='./content', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


# Function to denormalize the image
def denormalize(image):
    image = image * 0.5 + 0.5  # Scale to [0, 1]
    return image

# CIFAR-100 class names
class_names = train_dataset.dataset.classes

# Get a batch of images and labels from the training DataLoader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Convert the images from tensors to NumPy arrays
images = images.numpy()

# Denormalize the images
images = denormalize(images)

# Plot the images with their labels
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        # Transpose the image from (C, H, W) to (H, W, C) for Matplotlib
        img = np.transpose(images[i], (1, 2, 0))
        ax.imshow(img)
        ax.set_title(class_names[labels[i]], fontsize=8)
        ax.axis('off')
    else:
        ax.axis('off')  # Hide empty subplots

plt.tight_layout()
plt.show()

import timeit
from tqdm.notebook import tqdm

# init model
model = ViT(n_encoders, latent_size, n_classes, dropout).to(device)
# COST
criterion = nn.CrossEntropyLoss()

# Optimizer
opt = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

train_lossL = []
val_lossL = []
accuracy = []
start = start = timeit.default_timer()
# epoch loop
for epoch in tqdm(range(epochs), desc='Epochs'):
    model.train()
    train_loss = 0.0
    correct = 0
    total_samples = 0
    val_loss = 0.0

    # train loop
    for batch, (image, label) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epoch}', leave=False):

        image, label = image.to(device), label.to(device)
        # print(image.shape)  # Should be (batch_size, channels, height, width)
        opt.zero_grad() # Zero out gradient
        preds = model(image)
        loss = criterion(preds, label) # calc loss
        loss.backward() # backprop
        opt.step() # step
        train_loss += loss.item()

    # store loss
    Epoch_loss = train_loss / len(train_loader)
    train_lossL.append(Epoch_loss)

    # val loop
    model.eval()
    with torch.no_grad():
        for batch, (image, label) in enumerate(val_loader):
            image, label = image.to(device), label.to(device)
            preds = model(image)
            loss = criterion(preds, label)
            val_loss += loss.item()

            # get labels argmax for accurqcy
            y_pred_label = torch.argmax(preds, dim=1)
            correct += (y_pred_label == label).sum().item()
            total_samples += label.size(0)

    epoch_val_loss = val_loss / len(val_loader)
    val_lossL.append(epoch_val_loss)
    epoch_accuracy = correct / total_samples
    accuracy.append(epoch_accuracy)

    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print(f"Valid Accuracy EPOCH {epoch+1}: {epoch_accuracy:.4f}")

stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")

labels = []
# ids = []
imgs = []
model.eval()
with torch.no_grad():
    for idx, sample in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        img = sample["image"].to(device)
        # ids.extend([int(i)+1 for i in sample["index"]])
        
        outputs = model(img)
        
        imgs.extend(img.detach().cpu())
        labels.extend([int(i) for i in torch.argmax(outputs, dim=1)])

plt.figure()
f, axarr = plt.subplots(2, 3)
counter = 0
for i in range(2):
    for j in range(3):
        axarr[i][j].imshow(imgs[counter].squeeze(), cmap="gray")
        axarr[i][j].set_title(f"Predicted {labels[counter]}")
        counter += 1

plt.imshow()

# save the model
torch.save(model.state_dict(), 'Vit_model.pth')

# load the model
model.load_state_dict(torch.load('Vit_model.pth'))

# infer and plot samples with saved model
model.eval()
with torch.no_grad():
    for idx, sample in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        img = sample["image"].to(device)
        outputs = model(img)
        predicted = torch.argmax(outputs, dim=1)
        plt.imshow(img.detach().cpu().squeeze(), cmap="gray")
        plt.title(f"Predicted {predicted.item()}")
        plt.show()

        