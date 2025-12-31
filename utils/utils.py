from typing import Optional, Callable, Union, Tuple, Any
import os, sys
import random
import time
import datetime
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Ensure the plots use seaborn's default style
sns.set_theme(style="darkgrid")


def tensor2image(tensor):
    """Convert a tensor to a numpy image array."""
    image = 127.5 * (tensor.cpu().detach().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

class Logger:
    def __init__(self, n_epochs, batches_epoch):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.fig, self.ax = plt.subplots()
        self.loss_history = {}

    def log(self, losses=None, images=None):
        """Log the losses and display images."""
        # Calculate time elapsed
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()
        sys.stdout.write(f'\rEpoch {self.epoch:03d}/{self.n_epochs:03d} [{self.batch:04d}/{self.batches_epoch:04d}] -- ')
        # Update loss tracking
        for loss_name, loss_value in losses.items():
            loss_val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            if loss_name not in self.losses:
                self.losses[loss_name] = loss_val
                self.loss_history[loss_name] = []
            else:
                self.losses[loss_name] += loss_val
            sys.stdout.write(f"{loss_name}: {self.losses[loss_name]/self.batch:.4f} ")
        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write(f'-- ETA: {datetime.timedelta(seconds=int(batches_left * self.mean_period / batches_done))}')
        # Display images using matplotlib
        if images is not None:
            for image_name, tensor in images.items():
                self._save_image(tensor2image(tensor), image_name)
        # Update losses plot at the end of each epoch
        if (self.batch % self.batches_epoch) == 0:
            self._plot_losses()
            self.epoch += 1
            self.batch = 1
        else:
            self.batch += 1

    def _plot_losses(self):
        """Plot losses using matplotlib."""
        for loss_name, loss_value in self.losses.items():
            avg_loss = loss_value / self.batches_epoch
            self.loss_history[loss_name].append(avg_loss)
        self.ax.clear()
        for loss_name, loss_values in self.loss_history.items():
            self.ax.plot(loss_values, label=loss_name)
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        plt.pause(0.01)

    def _save_image(self, image, image_name):
        """Save the generated image using matplotlib."""
        print(image_name)
        os.makedirs('output_images', exist_ok=True)
        if len(image.shape) == 4:
            for i in range(image.shape[0]):
                plt.imsave(f'output_images/{image_name}_{i}.png', image[i].transpose(1, 2, 0))
        else:
            plt.imsave(f'output_images/{image_name}.png', image.transpose(1, 2, 0))



class ReplayBuffer:
    """A buffer that stores previously generated images."""
    def __init__(self, max_size=50):
        assert max_size > 0, 'Buffer size must be greater than zero.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """Push new data to the buffer and return a sample."""
        to_return = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return, dim=0)

class LambdaLR:
    """Learning rate scheduler with linear decay."""
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """Compute the learning rate multiplier for the current epoch."""
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    """Initialize model weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)


def logits_to_labels(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    ''' accepts logits as multichannel input, returns long tensor of class indices
        Args:
            preds: multichannel logit predictions in shape (N,C,H,W) or (C,H,W)
            num_classes: number of channels C
            dims: The resize dimensions if applicable
        Returns
            long tensor of class labels of shape (1,H,W) or (N,1,H,W)
    '''
    is_batch = int(len(preds.shape) == 4)
    assert num_classes > 1, f"Number of classes must be greater than 1"
    if not (preds.shape[is_batch] == num_classes):
        raise ValueError(f"expected {num_classes}-channel input in shape (N,C,H,W) or (C,H,W), got shape {tuple(preds.shape)}")
    #util.get_all_debug(softmaxed, "softmaxed")
    # FIXME: might switch to sigmoid
    pred_indices = torch.argmax(torch.nn.functional.softmax(preds, dim=is_batch), dim=is_batch, keepdim=True)
    return pred_indices.to(dtype=torch.uint8)