import datetime
import json
import pathlib

import torch


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


def evaluate(model, val_loader):
    """Evaluate the model's performance."""
    outputs = []
    for i, batch in enumerate(val_loader):
        # if i % 1000 == 0:
        #     logger.info(f"{i} / {len(val_loader)}")

        # TODO: check instances
        outputs.append(model.validation_step(batch, expand=True))
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    """Train the model using gradient descent."""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = pathlib.Path(f"./saved_models/{current_time}")
    save_dir.mkdir(exist_ok=True, parents=True)

    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # logger.info("Training Phase...")
        model.train()
        for i, batch in enumerate(train_loader):
            # if i % 1000 == 0:
            #     logger.info(f"{i} / {len(train_loader)}")

            # TODO: check instances
            loss = model.training_step(batch, expand=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # logger.info("Validation phase...")
        with torch.no_grad():
            model.eval()
            result = evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)

        torch.save(model.state_dict(), save_dir / f"epoch{epoch}.pt")
        with open(save_dir / f"epoch{epoch}.json", "w") as file:
            json.dump(history[-1], file)
    return history
