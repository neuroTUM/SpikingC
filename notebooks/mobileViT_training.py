# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import random_split
import torchvision

# Additional Imports
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Dataset
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset

# Network
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils

# Set the seed for reproducibility of results
torch.manual_seed(0)

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as torchvision_transforms
import tonic
import tonic.transforms as transforms

from tqdm import tqdm



config = {
    # SNN
    "threshold1": 2.5,
    "threshold2": 8.0,
    "threshold3": 4.0,
    "beta": 0.5,
    "num_steps": 10,
    
    # SNN Dense Shape
    "dense1_input": 2312,
    "num_classes": 10,

    # Hyper Params
    "lr": 0.007,

    # Early Stopping
    "min_delta": 1e-6,
    "patience_es": 20,

    # Training
    "epochs": 1
}

import torch
import torch.nn as nn

from einops import rearrange

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
threshold = 1.0

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        #nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        #nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        print(x.shape)
        # Local representations
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2), input_channels=1):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(input_channels, channels[0], stride=2)
        self.conv1_lif = snn.Leaky(beta=beta,  threshold=threshold, spike_grad=spike_grad)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])
        self.conv2_lif = snn.Leaky(beta=beta,  threshold=threshold, spike_grad=spike_grad)

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
        self.lif_out = snn.Leaky(beta=beta,  threshold=threshold, spike_grad=spike_grad)

    def forward(self, input):
        
        _, mem_conv1 = self.conv1_lif.init_leaky()
        _, mem_conv2 = self.conv2_lif.init_leaky()
        spk_out, mem_out = self.lif_out.init_leaky()
        
        #mem_rec = []
        #spk_rec = []
        
        for step in range(input.shape[0]):
            x = input[step]
            
            x = self.conv1(x)
            x, mem_conv1 = self.conv1_lif(x, mem_conv1)
            x = self.mv2[0](x)

            x = self.mv2[1](x)
            x = self.mv2[2](x)
            x = self.mv2[3](x)      # Repeat

            x = self.mv2[4](x)
            x = self.mvit[0](x)

            x = self.mv2[5](x)
            x = self.mvit[1](x)

            x = self.mv2[6](x)
            x = self.mvit[2](x)
            x = self.conv2(x)
            x, mem_conv2 = self.conv2_lif(x, mem_conv2)
            
            

            x = self.pool(x).view(-1, x.shape[1])
            x = self.fc(x)
            spk_out, mem_out = self.lif_out(x, mem_out)
            
        return torch.stack(spk_out), torch.stack(mem_out)


def mobilevit_xxs(num_classes):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    snn_params = []
    return MobileViT((128, 128), dims, channels, num_classes=num_classes, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=config["patience_es"], min_delta=config["min_delta"]):
        # Early stops the training if validation loss doesn't improve after a given patience.
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            print(f"Earlystop {self.counter}/{self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        #print(data.shape)

        optimizer.zero_grad()
        print(data.shape)
        spike_out, _ = model(data)
        print(spike_out.shape)
        output = spike_out
        #output = spike_out.sum(dim=0)
        loss = criterion(output, targets)
        running_loss += loss.item()

        _, predicted_train = torch.max(output.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted_train == targets).sum().item()
        
        print(f"Train Loss: {loss.item():.2f}")
        
        acc = SF.accuracy_rate(spike_out, targets) 
        print(f"Accuracy: {acc * 100:.2f}%\n")

        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    return train_loss, train_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            spike_out, _ = model(data)
            output = spike_out.sum(dim=0)
            loss = criterion(output, targets)
            val_loss += loss.item()

            _, predicted_val = torch.max(output.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted_val == targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    return val_loss, val_accuracy

def accuracy(output, target):
    """Compute the accuracy of predictions.

    Args:
        output (torch.Tensor): The output logits or probabilities from the model.
        target (torch.Tensor): The ground truth labels.

    Returns:
        float: The accuracy as a percentage.
    """
    # Get the predicted classes by finding the index of the maximum value in the logits dimension
    preds = output.argmax(dim=1)
    # Calculate the number of correctly predicted samples
    correct = preds.eq(target).sum().item()
    # Calculate accuracy by dividing the number of correct predictions by the total number of samples
    acc = correct / target.size(0)
    return acc * 100  # Return accuracy as a percentage
 
if __name__ == "__main__":
    
    # Define sensor size for NMNIST dataset
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    # Define transformations
    # Note: The use of torch.from_numpy is removed as Tonic's transforms handle conversion.
    transform = tonic.transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=100),
        #transforms.ToFrame(sensor_size=sensor_size, time_window=100000),
        #transforms.ToImage(sensor_size=sensor_size),
        # torchvision.transforms.RandomRotation is not directly applicable to event data.
        # If rotation is needed, it should be done on the frames after conversion by ToFrame.
    ])

    # Load NMNIST datasets without caching
    trainset = tonic.datasets.DVSGesture(save_to='./tmp/data', transform=transform, train=True)
    testset = tonic.datasets.DVSGesture(save_to='./tmp/data', transform=transform, train=False)

    # Split trainset into training and validation datasets
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
    
    num_classes = len(set(trainset.targets))
    print(f"Number of classes: {num_classes}")

    # Create DataLoaders for training, validation, and testing
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True))
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True))

    # Fetch a single batch from the train_loader to inspect the shape
    #data, targets = next(iter(train_loader))
    #print(f"Data shape: {data.shape}")  # Example output: torch.Size([batch_size, timesteps, channels, height, width])
    #print(f"Targets shape: {targets.shape}")  # Example output: torch.Size([batch_size])
    print("tiki")
    
    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SNN(config).to(device)
    model = mobilevit_xxs(num_classes).to(device)

    # Optimizer and Loss Function
    optimizer = Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Early Stopping
    early_stopping = EarlyStopping(patience=config["patience_es"], min_delta=config["min_delta"])
    

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    best_val_accuracy = 0
    model_path = "best_SNN_model.pth"

    for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        tqdm.write(f"Epoch: {epoch + 1}, Training Loss: {train_loss:.5f}, Training Accuracy: {train_accuracy:.2f}%, "
                f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.2f}%\n")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"Saved model with improved validation accuracy: {val_accuracy:.2f}% \n")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            tqdm.write("\nEarly stopping triggered")
            break