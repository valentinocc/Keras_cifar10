
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class LRFinder(Callback):
    #adjuted callback from Lucas Anders at: https://github.com/LucasAnders1/LearningRateFinder/blob/master/lr_finder_callback.py
    #adjusted to geometrically increase by step size instead of linearly increase
    #adjusted to run on tensorflow.keras instead of plain ole' keras
    '''
    This callback implements a learning rate finder(LRF)
    The learning rate is constantly increased during training.
    On training end, the training loss is plotted against the learning rate.
    One may choose a learning rate for a model based on the given graph,
    selecting a value slightly before the minimal training loss.
    The idea was introduced by Leslie N. Smith in this paper: https://arxiv.org/abs/1506.01186
    # Example
        lrf = LRFinder(max_iterations=5000, base_lr = 0.0001, max_lr = 0.1)
        model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=[LRF])
    # Arguments
        max_iterations: training stops when max_iterations are reached
        base_lr: initial learning rate used in training
        max_lr: training stops when learning rate exceeds max_lr
        lr_step_size: for each batch, the learning rate is increased by
            lr_step_size
    '''

    def __init__(self, max_iterations=5000, base_lr=0.0001, max_lr=0.1, lr_step_size=1.05):
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.lr_step_size = lr_step_size
        self.losses = []
        self.lrs = []
        self.lr = base_lr

    def on_train_batch_end(self, batch, logs={}):
        iterations = logs.get('batch')
        if (iterations >= self.max_iterations or self.lr >= self.max_lr):
            self.model.stop_training = True
        self.lr = self.lr * self.lr_step_size
        K.set_value(self.model.optimizer.lr, self.lr)
        self.losses.append(logs.get('loss'))
        self.lrs.append(self.lr)

    def on_train_end(self, logs=None):
        plt.plot(self.lrs, self.losses)
        plt.show()
