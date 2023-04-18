import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
    
    
class DNM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(DNM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define neural maps layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        
        # Define activation function
        self.activation = nn.Tanh()
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = x.view(batch_size, -1, self.output_size)
        return x
    
    def dnm_loss(x, y):
    # Compute the topographic error between x and y
        x = x.view(x.shape[0], -1, x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-1])
        d = ((x[:, None, :, :] - y[None, :, :, :]) ** 2).sum(-1)
        t = d.argmin(-1)
        te = ((t[:, None, :] - t[None, :, :]) ** 2).sum(-1).mean()
        # Compute the quantization error between x and y
        qe = ((x[:, None, :, :] - y[None, :, :, :]) ** 2).mean()
        # Return the sum of the topographic error and the quantization error
        return te + qe
    
    
    def train_dnm(model, train_loader, val_loader, num_epochs, lr):
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Define learning rate scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        # Define loss history
        train_loss_history = []
        val_loss_history = []
        # Train the model
        for epoch in range(num_epochs):
            # Set model to train mode
            model.train()
            train_loss = 0
            for x in train_loader:
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                y = model(x)
                # Compute loss
                loss = dnm_loss(x, y)
                train_loss += loss.item()
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()
            # Compute average training loss
            train_loss /= len(train_loader)
            # Compute validation loss
                # Set model to evaluation mode
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x in val_loader:
                    # Forward pass
                    y = model(x)
                    # Compute loss
                    loss = dnm_loss(x, y)
                    val_loss += loss.item()
            # Compute average validation loss
            val_loss /= len(val_loader)
            # Append loss history
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            # Update learning rate scheduler
            lr_scheduler.step(val_loss)
        # Return the trained model and the loss history
        return model, train_loss_history, val_loss_history


############################################################################################################################
# makefile

Split the dataset into training and validation sets:

```python
# Split the dataset into training and validation sets
train_data = my_data[:800,:,:]
val_data = my_data[800:,:,:]
# Define batch size
batch_size = 32
# Create data loaders
train_loader = DataLoader(MyDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(MyDataset(val_data), batch_size=batch_size, shuffle=False)



# Define model parameters
input_size = my_data.shape[1] * my_data.shape[2]
output_size = 2
hidden_size = 128
num_layers = 3
num_epochs = 100
lr = 1e-3
# Initialize model
model = DNM(input_size, output_size, hidden_size, num_layers)
# Train model
trained_model, train_loss_history, val_loss_history = train_dnm(model, train_loader, val_loader, num_epochs, lr)
