import torch
import torch.nn.functional as f
from loguru import logger

from src.models.metrics import accuracy


class StackedCNNBaseline(torch.nn.Module):
    def __init__(self):
        super(StackedCNNBaseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 18, 3)
        self.fc1 = torch.nn.Linear(18 * 35 * 35, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        cur_x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv2(cur_x)), (2, 2))
        cur_x = f.max_pool2d(f.relu(self.conv3(cur_x)), (2, 2))
        cur_x = cur_x.view(x.shape[0], -1)
        cur_x = f.relu(self.fc1(cur_x))
        cur_x = f.relu(self.fc2(cur_x))
        cur_x = self.fc3(cur_x)
        return cur_x

    def training_step(self, batch):
        images, labels = batch

        out = torch.zeros(size=(images.shape[0], 2), device=self.device)
        for cur_image in images.permute(1, 0, 2, 3, 4):
            out += torch.div(self(cur_image), images.shape[1])  # Generate predictions
        loss = f.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch

        out = torch.zeros(size=(images.shape[0], 2), device=self.device)
        for cur_image in images.permute(1, 0, 2, 3, 4):
            out += torch.div(self(cur_image), images.shape[1])  # Generate predictions
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