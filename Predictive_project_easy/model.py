
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import math
from torchmetrics.regression import MeanAbsoluteError


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=150, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):  # Removed input_lengths from arguments
        # Assuming all sequences are of the same length, thus not using pack_padded_sequence
        lstm_out, _ = self.lstm(input_seq)
        # Directly use the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out)
        return predictions

def train_and_validate_model(model, train_loader, valid_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    train_rmse = []
    valid_rmse = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # Adjusted to match the updated forward method
            y_pred = torch.clamp(y_pred, min=0)  # Clipping prediction to a minimum of 0
            loss = loss_function(y_pred, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seq.size(0)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for seq, labels in valid_loader:
                y_pred = model(seq).squeeze()
                y_pred = torch.clamp(y_pred, min=0)
                loss = loss_function(y_pred, labels.float())
                valid_loss += loss.item() * seq.size(0)

        train_rmse_epoch = np.sqrt(train_loss / len(train_loader.dataset))
        valid_rmse_epoch = np.sqrt(valid_loss / len(valid_loader.dataset))
        train_rmse.append(train_rmse_epoch)
        valid_rmse.append(valid_rmse_epoch)

        print(f'Epoch {epoch+1}: Train RMSE: {train_rmse_epoch}, Validation RMSE: {valid_rmse_epoch}')

    return train_rmse, valid_rmse

def evaluate_model(model, test_loader):
    model.eval()
    loss_function = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for seq, labels in test_loader:
            y_pred = model(seq).squeeze()
            y_pred = torch.clamp(y_pred, min=0)
            loss = loss_function(y_pred, labels.float())
            test_loss += loss.item() * seq.size(0)

    avg_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss}')




def compute_s_score_torch(rul_true, rul_pred):
    """
    Compute the S-Score for PyTorch tensors.
    Both rul_true and rul_pred should be PyTorch tensors.
    """
    diff = rul_pred - rul_true
    s_score = torch.sum(torch.where(diff < 0, torch.exp(-diff / 13) - 1, torch.exp(diff / 10) - 1))
    return s_score

def plot_rmse(train_rmse, valid_rmse):
    plt.figure(figsize=(10, 5))
    plt.plot(train_rmse, label='Training RMSE')
    plt.plot(valid_rmse, label='Validation RMSE')
    plt.title('Training and Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

def predict_and_evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    
    with torch.no_grad():  # No need to track gradients for evaluation
        for seq, labels in test_loader:
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq).squeeze()
            predictions.extend(y_pred.tolist())  # Store predictions
            actuals.extend(labels.tolist())  # Store actual values
    
    # Calculate MAE
    mae = mean_absolute_error(actuals, predictions)
    print(f'Test MAE: {mae}')
    return predictions, actuals, mae



class ComplexLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, output_size=1, num_layers=2, dropout_rate=0.5, bidirectional=False):
        super(ComplexLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define a more complex LSTM layer with optional bidirectionality and dropout
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, 
                            dropout=dropout_rate if num_layers > 1 else 0, 
                            bidirectional=bidirectional, batch_first=True)
        
        # Adjusting the input feature size for the linear layer depending on bidirectionality
        linear_input_size = hidden_layer_size * 2 if bidirectional else hidden_layer_size
        
        # Additional linear layers for complexity
        self.linear_layers = nn.Sequential(
            nn.Linear(linear_input_size, linear_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(linear_input_size // 2, output_size)
        )

    def forward(self, input_seq):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        
        # Forward pass through LSTM layer
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        
        # Take the output of the last LSTM layer
        if self.bidirectional:
            lstm_out = lstm_out[:, -1, :]
        else:
            lstm_out = lstm_out[:, -1, :]
        
        # Pass through linear layers
        predictions = self.linear_layers(lstm_out)
        
        return predictions
 
def evaluate_model(model, test_loader):  # noqa: F811
    model.eval()
    loss_function = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for seq, labels in test_loader:
            y_pred = model(seq).squeeze()
            y_pred = torch.clamp(y_pred, min=0)
            loss = loss_function(y_pred, labels.float())
            test_loss += loss.item() * seq.size(0)
    mean_absolute_error = MeanAbsoluteError()   
    mae = mean_absolute_error(y_pred.int(), labels.int())


    avg_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {math.sqrt(avg_loss)}')
    print(f'Test MAE: {mae}')