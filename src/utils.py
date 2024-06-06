import datetime
import json
import pathlib

import torch
from loguru import logger


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device."""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device."""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def evaluate(model, val_loader, time_dimension=None):
    """Evaluate the model's performance."""
    outputs = []
    for i, batch in enumerate(val_loader):
        outputs.append(model.validation_step(batch, time_dimension=time_dimension))
    return model.validation_epoch_end(outputs)


def predict(model, test_loader, time_dimension=None):
    """Make predictions with the model."""
    outputs = []
    for _, batch in enumerate(test_loader):
        outputs.append(model.prediction_step(batch, time_dimension=time_dimension))
    return torch.cat(outputs, dim=0)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fit(epochs, lr, model, train_loader, val_loader, test_loader=None, time_dimension=None, opt_func=torch.optim.Adam):
    """Train the model using gradient descent."""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = pathlib.Path(f"./saved_models/{current_time}")
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving to {save_dir}")

    history_val = []
    history_test = []

    optimizer = opt_func(model.parameters(), lr)
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            loss = model.training_step(batch, time_dimension=time_dimension)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            result_val = evaluate(model, val_loader, time_dimension=time_dimension)
            model.epoch_end(epoch, result_val)
            history_val.append(result_val)
            with open(save_dir / f"epoch{epoch}_val.json", "w") as file:
                json.dump(history_val[-1], file)

            if test_loader is not None:
                result_test = evaluate(model, test_loader, time_dimension=time_dimension)
                history_test.append(result_test)
                
                with open(save_dir / f"epoch{epoch}_test.json", "w") as file:
                    json.dump(history_test[-1], file)

        torch.save(model.state_dict(), save_dir / f"epoch{epoch}.pt")

        if early_stopper.early_stop(result_val["val_loss"]):
            logger.info(f"Early stop at epoch {epoch}!")
            break

    return history_val