import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM model class

    Attributes 
    ----------
    input_size: int
        size of the input vector
    output_size: int
        size of the prediction vector
    hidden_size: int
        size of the hidden state for each LSTM layer
    num_layers: int
        number of recurrent LSTM layers

    Methods 
    -------
    forward(x) model forward pass, returns output tensor
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """model forward pass"""
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out
    