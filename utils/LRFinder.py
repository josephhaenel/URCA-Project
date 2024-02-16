import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class LRFinder(tf.keras.callbacks.Callback):
    """
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    """

    def __init__(self, start_lr=1e-7, end_lr=10, num_iter=100, stop_multiplier=None):
        super(LRFinder, self).__init__()
        
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.stop_multiplier = stop_multiplier
        self.lrs = []
        self.losses = []
        self.best_loss = 1e9
        self.iteration = 0
        self.avg_loss = 0
        self.beta = 0.98  # For the exponentially weighted average

    def on_train_begin(self, logs=None):
        self.lrs = []
        self.losses = []
        self.iteration = 0
        self.best_loss = 1e9
        self.avg_loss = 0
        tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

    def on_batch_end(self, batch, logs=None):
        lr = self._get_rate()
        self.lrs.append(lr)

        loss = logs['loss']
        self.iteration += 1

        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** self.iteration)

        if self.iteration > 1 and smoothed_loss > self.best_loss * self.stop_multiplier:
            self.model.stop_training = True
            return

        if smoothed_loss < self.best_loss or self.iteration == 1:
            self.best_loss = smoothed_loss

        self.losses.append(smoothed_loss)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def _get_rate(self):
        """Calculates the learning rate at each iteration."""
        x = self.iteration / self.num_iter
        return self.start_lr + (self.end_lr - self.start_lr) * x

    def plot_loss(self):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.title('Learning rate finder')
        plt.show()