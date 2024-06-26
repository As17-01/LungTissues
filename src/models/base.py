import torch
import torch.nn.functional as f
from loguru import logger
from torcheval.metrics import BinaryAUROC

from src.models.metrics import accuracy


class BaseModel(torch.nn.Module):
    def training_step(self, batch, time_dimension=None):
        images, labels = batch
        if time_dimension is not None:
            images = torch.swapaxes(images, time_dimension, 2)
        images = images.to(torch.float32)

        # if expand:
        #     labels = labels.repeat_interleave(images.shape[1])
        #     labels = labels.unsqueeze(1).to(torch.float32)
        #     images = images.view(-1, *images.shape[-3:])
        # else:

        labels = labels.to(torch.float32)
        out = self(images)  # Generate predictions

        loss = f.binary_cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch, time_dimension=None):
        images, labels = batch
        if time_dimension is not None:
            images = torch.swapaxes(images, time_dimension, 2)
        images = images.to(torch.float32)

        # if expand:
        #     labels = labels.repeat_interleave(images.shape[1])
        #     labels = labels.unsqueeze(1).to(torch.float32)
        #     images = images.view(-1, *images.shape[-3:])
        # else:

        labels = labels.to(torch.float32)
        out = self(images)  # Generate predictions

        loss = f.binary_cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss, "val_acc": acc, "preds": out, "labels": labels}

    def prediction_step(self, batch, time_dimension=None):
        images, _ = batch
        if time_dimension is not None:
            images = torch.swapaxes(images, time_dimension, 2)
        images = images.to(torch.float32)

        out = self(images)  # Generate predictions
        return out

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        batch_preds = torch.cat([x["preds"] for x in outputs])
        batch_labels = torch.cat([x["labels"] for x in outputs])
        roc_auc_metric = BinaryAUROC()
        roc_auc_metric.update(torch.squeeze(batch_preds), torch.squeeze(batch_labels))
        roc_auc = roc_auc_metric.compute()

        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item(), "val_roc_auc": roc_auc.item()}

    def epoch_end(self, epoch, result, suffix: str = "val"):
        s = []
        s.append(f"Epoch [{epoch}]")
        s.append(f"{suffix}_loss: {result['val_loss']:.4f}")
        s.append(f"{suffix}_acc: {result['val_acc']:.4f}")
        s.append(f"{suffix}_roc_auc: {result['val_roc_auc']:.4f}")
        logger.info(" ".join(s))
