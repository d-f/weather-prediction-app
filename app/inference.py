import torch
from pathlib import Path
import utils


def torch_device():
    """return torch device for CPU"""
    return torch.device("cpu")


def prepare_input(input: list, min_val: float, max_val: float) -> torch.tensor:
    """
    normalize input and convert to tensor
    
    keyword arguments:
    input   -- input list 
    min_val -- minimum value from the dataset
    max_val -- maximum value from the dataset
    """
    input = torch.tensor(input)
    input = utils.data_utils.normalize(min_val=min_val, max_val=max_val, x=input)
    return input


def prepare_prediction(pred: torch.tensor, min_val: float, max_val: float) -> float:
    """
    undos normalization

    keyword arguments:
    pred    -- prediction
    min_val -- minimum value from dataset
    max_val -- maximum value from dataset
    """
    range_val = max_val - min_val
    pred *= range_val
    pred += min_val
    return pred.numpy().tolist()


def predict(input):
    """returns model prediction for a given input"""
    BASE_DIR = Path(__file__).resolve(strict=True).parent

    MODEL_PATH = BASE_DIR.joinpath("model_1.pth.tar")
    DATA_PREP_JSON = BASE_DIR.joinpath("data_prep.json")

    data_prep_json = utils.data_utils.read_json(DATA_PREP_JSON)
    min_val = data_prep_json["min"]
    max_val = data_prep_json["max"]
    device = torch_device()
    model = utils.model_utils.model(input_size=12, output_size=24, hidden_size=256, num_layers=3)
    model = utils.model_utils.load_model(
        model=model, 
        weight_path=Path(MODEL_PATH),
        cpu=True
        )
    input = prepare_input(input=input, min_val=min_val, max_val=max_val)
    
    input.to(device)
    model.to(device)

    pred = model(input)    
    return prepare_prediction(pred.detach(), min_val=min_val, max_val=max_val)
