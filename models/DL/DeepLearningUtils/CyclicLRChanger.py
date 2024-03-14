from keras.callbacks import Callback
import numpy as np
from keras import backend as K


class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1.):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        if self.mode == 'triangular2':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        if self.mode == 'exp_range':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** (self.clr_iterations))

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            self.clr_iterations = 0
        else:
            self.clr_iterations = self.trn_iterations
        
        self.trn_iterations = 0
        self.history.setdefault('lr', []).append(self.base_lr)
        self.history.setdefault('iterations', []).append(self.clr_iterations)

        self.set_lr(self.model, self.base_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(self.clr())
        self.history.setdefault('iterations', []).append(self.clr_iterations)

        self.set_lr(self.model, self.clr())

    def set_lr(self, model, lr):
        K.set_value(model.optimizer.lr, lr)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.clr()

