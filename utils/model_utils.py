import lstm
import torch
from torch.optim import Adam
from pathlib import Path
from typing import Type, Dict


def torch_device() -> torch.device:
    """returns torch device"""
    return torch.device("cuda")


def optimizer(model: Type[lstm.LSTM], learning_rate: float) -> Type[Adam]:
    """
    returns optimizer algorithm

    keyword arguments:
    model         -- LSTM from /lstm/lstm.py
    learning_rate -- float value to weight gradient values by
    """
    return Adam(model.parameters(), lr=learning_rate)


def loss() -> Type[torch.nn.MSELoss]:
    """returns torch mean squared error loss"""
    return torch.nn.MSELoss()


def model(input_size: int, output_size: int, hidden_size: int, num_layers: int) -> Type[lstm.LSTM]:
    """
    returns stacked lstm model with given input size, output size, and number of LSTM layers

    keyword arguments:
    input_size  -- size of the input vector 
    output_size -- size of the prediction vector
    hidden_size -- number of features in each hidden state
    num_layers  -- number of recurrent layers
    """
    return lstm.LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers)


def save_checkpoint(state: Dict, filepath: Path) -> None:
    """
    saves the model state dictionary to a .pth.tar tile

    keyword arguments:
    state    -- dictionary object with model parameters
    filepath -- Path object of location to save model file to
    """
    print("saving...")
    torch.save(state, filepath)


def load_model(weight_path: Path, model: type[lstm.LSTM], cpu=False):
    """
    loads all parameters of a model

    keyword arguments:
    weight_path -- path to model .pth.tar file
    model       -- LSTM from /lstm/lstm.py
    """
    if cpu:
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        return model
    else:
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint)
        return model
