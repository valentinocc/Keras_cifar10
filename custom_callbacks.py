
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

class LR_pattern_smith(Callback):
    '''
    The learning rate is linearly increased from base_lr to max_lr, then linear decreased back to base_lr, and then
    held constant at a low learning rate (min_lr) for the final epochs (Around 20-35% of epochs)
    The idea was introduced by Leslie N. Smith in this paper: https://arxiv.org/abs/1506.01186
    # Example
        lra = LR_adjuster(15, min_lr = 0.002, max_lr = 0.1, base_lr = 0.04)
        model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=[lra])
    # Arguments
        epochs: the amount of epochs used to train the neural network
        base_lr: initial learning rate used in training
        max_lr: the highest learning rate to be used in training, the learning rate will decrease after reaching this rate
                this learning rate should be set using methods discussed in Smith's paper https://arxiv.org/pdf/1803.09820.pdf
        min_lr: the learning rate to be used for the last 20-30% of epochs
    '''

    def __init__(self, epochs, min_lr = 0.0015, base_lr=0.01, max_lr=0.1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = 0.0015
        self.epochs_max_point = (epochs - 5) / 2
        self.lr_step_size = (max_lr - base_lr) / self.epochs_max_point
        self.lrs = []
        self.lr = base_lr
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs={}):

        if (epoch < self.epochs_max_point):
            self.lr = self.lr + self.lr_step_size
        elif (epoch >= self.epochs_max_point and epoch < self.epochs_max_point * 2):
            self.lr = self.lr - self.lr_step_size
        else:
            self.lr = self.min_lr

        K.set_value(self.model.optimizer.lr, self.lr)
        self.lrs.append(self.lr)
    
    def on_train_end(self, logs=None):
        plt.plot( np.arange(self.epochs), self.lrs)
        plt.show
        print(self.lrs)


class LR_adjuster(Callback):
    '''
    The learning rate is linearly increased from base_lr to max_lr, then linear decreased back to base_lr, and then
    held constant at a low learning rate (min_lr) for the final epochs (Around 20-35% of epochs)
    The idea was introduced by Leslie N. Smith in this paper: https://arxiv.org/abs/1506.01186
    # Example
        lra = LR_adjuster(15, min_lr = 0.002, max_lr = 0.1, base_lr = 0.04)
        model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=[lra])
    # Arguments
        epoch_switch: the epoch on which the base_lr is switched to the final_lr
        base_lr: initial learning rate used in training
        final_lr: the second learning rate to be used
    '''

    def __init__(self, epoch_switch, base_lr=0.002, final_lr=0.0002):
        self.final_lr = final_lr
        self.lrs = []
        self.lr = base_lr
        self.epoch_switch = epoch_switch

    def on_epoch_end(self, epoch, logs={}):

        if (epoch == self.epoch_switch):
            self.lr = self.final_lr

        K.set_value(self.model.optimizer.lr, self.lr)
        self.lrs.append(self.lr)

    def on_train_end(self, logs=None):
        plt.plot( np.arange(self.epochs), self.lrs)
        plt.show
        print(self.lrs)
