import torch
import torch.nn.functional as f
from loguru import logger

from src.models.metrics import accuracy


class CNNBaseline(torch.nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 72 * 72, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        slide_size = x.shape[1]

        scores = torch.zeros(size=(batch_size, 2))
        for cur_x in x.permute(1, 0, 2, 3, 4):
            cur_x = f.max_pool2d(f.relu(self.conv1(cur_x)), (2, 2))
            cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), 2)
            cur_x = cur_x.view(batch_size, -1)
            cur_x = f.relu(self.fc1(cur_x))
            cur_x = f.relu(self.fc2(cur_x))
            cur_x = self.fc3(cur_x)

            scores += torch.div(cur_x, slide_size)
        scores = f.sigmoid(scores)
        return scores

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = f.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = f.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        logger.info(
            "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["val_loss"], result["val_acc"])
        )
