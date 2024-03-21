import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils
import argparse
from pathlib import Path
import torch
from typing import Type
from lstm import LSTM


def parse_argparser() -> argparse.Namespace:
    """parses command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_size", type=int, default=24)
    parser.add_argument("-input_size", type=int, default=12)
    parser.add_argument("-hidden_size", type=int, default=1024)
    parser.add_argument("-dataset_json", type=Path, default=Path("C:\\personal_ML\\weather_prediction_app\\normalized_dataset.json"))
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-lr", type=float, default=1e-6)
    parser.add_argument("-num_layers", type=int, default=32)
    parser.add_argument("-num_epochs", type=int, default=256)
    parser.add_argument("-model_save_name",  type=str, default="model_10.pth.tar")
    parser.add_argument("-result_dir", type=Path, default=Path("C:\\personal_ML\\weather_prediction_app\\results\\"))
    parser.add_argument("-patience", type=int, default=5)
    return parser.parse_args()


class TSDataset():
    """
    creates a time series dataset

    keyword arguments:
    dataset_json -- path to the JSON file containing 
        "train", "validation", and "test" keys and
        dictionaries with "input" and "output" pairs as 
        values

    mode -- one of {"train", "val", "test"} in order to 
        access correct dataset partition
    """
    def __init__(self, dataset_json: Path, mode: str):
        self.mode = mode
        self.dataset = utils.data_utils.read_json(dataset_json)[self.mode]

    def __len__(self) -> int:
        return len(self.dataset["input"])
    
    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        input_tensor = self.dataset["input"][idx]
        output_tensor = self.dataset["output"][idx]
        return torch.tensor(input_tensor), torch.tensor(output_tensor)


def dataloaders(
        dataset_json_path: Path, 
        batch_size: int
        ) -> tuple[torch.utils.data.DataLoader]:
    """
    returns PyTorch DataLoaders

    keyword arguments:
    dataset_json_path -- Path object to dataset JSON file
    batch_size        -- number of instances in each dataloader output
    """
    train_ds = TSDataset(dataset_json=dataset_json_path, mode="train")
    val_ds = TSDataset(dataset_json=dataset_json_path, mode="val")
    test_ds = TSDataset(dataset_json=dataset_json_path, mode="test")

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def validate(
        model: Type[LSTM], 
        val_loader: Type[DataLoader], 
        device: torch.device, 
        loss_fn: Type[torch.nn.MSELoss]
        ) -> torch.tensor:
    """
    performs cross-validation on model with validation dataloader
    returns loss on validation set

    keyword arguments:
    model      -- LSTM from /lstm/lstm.py 
    val_loader -- validation dataloader
    device     -- torch device
    loss_fn    -- Mean Squared Error torch.nn.MSEloss
    """
    model.eval()
    val_loss = 0
    for batch_in, batch_out in tqdm(val_loader, desc="Validating"):
        pred = model(batch_in.to(device))
        loss = loss_fn(pred, batch_out.to(device))
        val_loss += loss
    val_loss /= len(val_loader)            
    return val_loss


def train(
        train_loader: Type[DataLoader], 
        val_loader: Type[DataLoader], 
        model: Type[LSTM], 
        device: torch.device, 
        loss_fn: Type[torch.nn.MSELoss], 
        optimizer: Type[torch.optim.Adam], 
        num_epochs: int, 
        model_save_name: str, 
        patience: int, 
        result_dir: Path
        ) -> tuple[list, list]:
    """
    trains LSTM on train dataset
    returns list containing train and val loss per epoch

    keyword arguments:
    train_loader    -- DataLoader for train dataset
    val_loader      -- DataLoader for validation dataset
    model           -- LSTM model
    device          -- torch device 
    loss_fn         -- Mean Squared Error torch.nn.MSEloss
    optimizer       -- torch Adam
    num_epochs      -- number of training iterations through the train dataset
    model_save_name -- name of the model to save, include .pth.tar
    patience        -- number of epochs past validation loss improving to continue training
    result_dir      -- Path to model and result file save location 
    """
    model.to(device)
    train_loss_list = []
    val_loss_list = []
    best_val_loss = np.inf
    patience_counter = 0
    for epoch_idx in range(num_epochs):
        model.train()
        if patience_counter == patience:
            break
        else:
            tr_epoch_loss = 0
            for batch_in, batch_out in tqdm(train_loader, desc="Training"):
                pred = model(batch_in.to(device))
                train_loss = loss_fn(pred, batch_out.to(device))
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                tr_epoch_loss += train_loss
            tr_epoch_loss /= len(train_loader)
            train_loss_list.append((epoch_idx + 1, float(tr_epoch_loss.cpu().detach())))
            
            val_loss = validate(model=model, val_loader=val_loader, device=device, loss_fn=loss_fn)
            val_loss_list.append((epoch_idx+1, float(val_loss.cpu().detach())))

            print(f"epoch {epoch_idx+1} train loss [{tr_epoch_loss}], val loss [{val_loss}]")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                utils.model_utils.save_checkpoint(state=model.state_dict(), filepath=result_dir.joinpath(model_save_name))
            else:
                patience_counter += 1                    

    return train_loss_list, val_loss_list


def test(
        model: Type[LSTM], 
        model_save_name: str, 
        test_loader: Type[DataLoader], 
        device: torch.device, 
        loss_fn: torch.nn.MSELoss, 
        save_dir: Path
        ) -> float:
    """
    evaluates model performance on test dataset

    keyword arguments:
    model           -- LSTM from /lstm/lstm.py
    model_save_name -- name to reload best saved model, including .pth.tar
    test_loader     -- DataLoader for test dataset
    device          -- torch device
    loss_fn         -- Mean Squared Error torch.nn.MSEloss
    save_dir        -- Path to save test result file to
    """
    model = utils.model_utils.load_model(model=model, weight_path=Path(save_dir).joinpath(model_save_name))
    test_loss = 0
    for batch_in, batch_out in tqdm(test_loader, desc="Testing"):
        pred = model(batch_in.to(device))
        loss = loss_fn(pred, batch_out.to(device))
        test_loss += loss
    test_loss /= len(test_loader)  
    print("test loss:", test_loss)          
    return float(test_loss.cpu().detach())


def main():
    args = parse_argparser()
    device = utils.model_utils.torch_device()
    loss = utils.model_utils.loss()
    model = utils.model_utils.model(input_size=args.input_size, output_size=args.output_size, num_layers=args.num_layers, hidden_size=args.hidden_size)
    optimizer = utils.model_utils.optimizer(model=model, learning_rate=args.lr)
    train_loader, val_loader, test_loader = dataloaders(args.dataset_json, batch_size=args.batch_size)
    train_loss, val_loss = train(
        train_loader=train_loader, 
        val_loader=val_loader, 
        model=model, 
        device=device, 
        loss_fn=loss, 
        optimizer=optimizer, 
        num_epochs=args.num_epochs,
        result_dir=args.result_dir,
        patience=args.patience,
        model_save_name=args.model_save_name
        )
    test_loss = test(model=model, test_loader=test_loader, device=device, loss_fn=loss, model_save_name=args.model_save_name, save_dir=args.result_dir)
    utils.data_utils.save_train_results(
        train_loss=train_loss, 
        val_loss=val_loss, 
        result_fp=args.result_dir.joinpath(args.model_save_name.rpartition(".pth.tar")[0]+"_train_results.json")
        )
    utils.data_utils.save_experiment(
        result_fp=args.result_dir.joinpath(args.model_save_name.rpartition(".pth.tar")[0]+"_experiment.csv"),
        batch_size=args.batch_size, learning_rate=args.lr, num_layers=args.num_layers, 
        hidden_size=args.hidden_size, input_size=args.input_size, output_size=args.output_size, test_loss=test_loss, model_save_name=args.model_save_name
    )

if __name__ == "__main__":
    main()
