
import types
import math
from torch import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim.optimizer import Optimizer

class StreamingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def streamavg(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        if len(self.values) <= self.window_size and len(self.values)>1:
            return float(self.sum) / len(self.values)
            
        return float(self.sum) / len(self.values)

class GreedyLR:
    """Reduce learning rate when a metric has stopped improving and
    Increase learning rate when a metric is improving. This scheduler
    greedily changes learning rate based on change in a metrics quantity
    and if no improvement is seen for a 'patience' number of epochs, the
    learning rate is reduced. Conversely if improvement is seen for a
    'patience' number of epochs, learning rate is increased.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            changed. new_lr = lr * factor. Default: 0.99.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        warmup (int): Number of epochs to wait before resuming
            normal operation after lr has been increased. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        max_lr (float or list): A scalar or a list of scalars. An
            upped bound on the learning rate of all param groups
            or each group respectively. Default: 1.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        reset_start (int): Runs the _reset function at a particular epoch,
            which then resets the cool down and warm up timer
        smooth (bool): Runs a streaming average on the LR based on window size
    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = GreedyLR(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.95, patience=10,
                 threshold=1e-6, threshold_mode='abs', cooldown=0, warmup=0,
                 min_lr=1e-3, max_lr=1, eps=1e-8, verbose=False, smooth=False, 
                 window_size=50, reset_start=500):
        
        
            
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
            self.max_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.warmup = warmup # ADDED
        self.cooldown_counter = 0
        self.warmup_counter = 0 #ADDED
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.num_good_epochs = None #ADDED
        self.mode_worse = None  # the worse value for the chosen mode
        self.mode_better = None  # ADDED : the better value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self.smooth = smooth
        self.reset_start = reset_start
        self.reset_start_original = reset_start
        if self.smooth:
            self.sa = StreamingAverage(window_size)
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()
        

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.reset_start = self.reset_start_original
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        
        self.warmup_counter = 0 #ADDED
        self.num_good_epochs = 0 #ADDED

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        
        if self.smooth:
            current = self.sa.streamavg(current)
        
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.num_good_epochs += 1 #ADDED
        else:
            self.num_bad_epochs += 1
            self.num_good_epochs = 0 #ADDED

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
            
        if self.in_warmup:
            self.warmup_counter -= 1
            self.num_good_epochs = 0  # ignore any good epochs in warmup

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            
        if self.num_good_epochs > self.patience: #ADDED
            self._increase_lr(epoch) #TODO
            self.warmup_counter = self.warmup
            self.num_good_epochs = 0
        
        if self.reset_start == 0:
            self._reset()
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        if len(set(self._last_lr)) == 1:
            # All at lower bound, try resetting
            self.reset_start -= 1
        


    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                
                param_group['lr'] = new_lr
                
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
                    
    def _increase_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = min(old_lr / self.factor, self.max_lrs[i])
            if new_lr - old_lr > self.eps:
                    
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
                    
    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
    
    @property # ADDED
    def in_warmup(self):
        return self.warmup_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
