import torch
import torch.nn as nn
import utils

class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, criterion2, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.criterion2 = criterion2
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            
            test = utils.one_hot(4, labels.reshape(labels.shape[0], 1, 200, 200))
            test = test.permute(0, 3, 1, 2)

            # 全局平均池化
            global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            test_pooled = global_avg_pool(test)

            # Forward propagation
            (outputs, ouputvec) = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)
            loss2 = self.criterion2(ouputvec, test_pooled)

            total_loss = loss + loss2

            # Backpropagation
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
