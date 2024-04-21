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
        # if i % 1000 == 0:
        #     logger.info(f"{i} / {len(val_loader)}")

        # TODO: check instances
        outputs.append(model.validation_step(batch, time_dimension=time_dimension))
    return model.validation_epoch_end(outputs)


def predict(model, test_loader, time_dimension=None):
    """Make predictions with the model."""
    outputs = []
    for _, batch in enumerate(test_loader):
        # TODO: check instances
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


def fit(epochs, lr, model, train_loader, val_loader, time_dimension=None, opt_func=torch.optim.Adam):
    """Train the model using gradient descent."""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = pathlib.Path(f"./saved_models/{current_time}")
    save_dir.mkdir(exist_ok=True, parents=True)

    history = []
    optimizer = opt_func(model.parameters(), lr)
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    for epoch in range(epochs):
        # logger.info("Training Phase...")
        model.train()
        for i, batch in enumerate(train_loader):
            # if i % 1000 == 0:
            #     logger.info(f"{i} / {len(train_loader)}")

            # TODO: check instances
            loss = model.training_step(batch, time_dimension=time_dimension)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # logger.info("Validation phase...")
        with torch.no_grad():
            model.eval()
            result = evaluate(model, val_loader, time_dimension=time_dimension)
            model.epoch_end(epoch, result)
            history.append(result)

        torch.save(model.state_dict(), save_dir / f"epoch{epoch}.pt")
        with open(save_dir / f"epoch{epoch}.json", "w") as file:
            json.dump(history[-1], file)

        if early_stopper.early_stop(result["val_loss"]):
            logger.info(f"Early stop at epoch {epoch}!")
            break
    return history
