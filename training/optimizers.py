import math
from typing import List
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, SGD, Nadam, RMSprop
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import numpy as np



optimizer_list = [
    "adadelta",
    "adagrad",
    "adam",
    "adam_amsgrad",
    "sgd",
    "sgd_momentum",
    "sgd_nesterov",
    "nadam",
    "rmsprop",
    "radam",
]

LEARNING_SCHEDULE_LIST = [
    "",
    "exponential_decay",
    "step_decay",
    "step_decay_schedule",
    "polynomial_decay",
    "inverse_time_decay",
    "cosine_decay",
    "lr_reduce_on_plateau",
    "explicit_schedule",
    "one_cycle",
    "flat_and_anneal",
    "lr_finder",
]

_END_LEARNING_RATE = 0.0000001

arguments = [
    [str, 'name', 'sgd_nesterov', 'optimized to be used', lambda x: x.lower() in optimizer_list],
    [float, 'momentum', 0.9, 'used when optimizer name is specified as sgd_momentum'],
    [float, "lr", 0.0001, 'initial learning rate', lambda x: x > 0],
    [float, "weight_decay", 0.0001, 'weight of l2 regularization', lambda x: x > 0],
    [float, "clipnorm", 0, 'if different than zero then use gradient norm clipping'],
    [float, "clipvalue", 0, 'if different than zero then use gradient value clipping'],
    [bool, "lookahead", False, "whether to use lookahead with the optimizer"],
    [int, "sync_period", 6, "Used only in lookahead. It is the synchronization period"],
    ['namespace', 'lr_decay_strategy', [
        [bool, 'activate', True, 'if true then use this callback'],
        ['namespace', 'lr_params', [
            [str, 'strategy', 'lr_reduce_on_plateau', 'learning rate decay schedule', lambda x: x.lower() in LEARNING_SCHEDULE_LIST],
            [int, 'power', 5, 'used only in polynomial_decay, determines the nth degree polynomial'],
            [float, 'alpha', 0.01, 'used only in cosine decay, Minimum learning rate value as a fraction of initial_learning_rate. '],
            [int, 'patience', 10, 'used only in lr_reduce_on_plateau, if validation loss doesn\'t improve for this number of epoch, then reduce the learning rate'],
            [float, 'decay_rate', 0.5, 'used step_decay, step_decay_schedule, inverse_time_decay, lr_reduce_on_plateau, determines the factor to drop the lr'],
            [float, 'min_lr', 0.00001, 'used in lr_reduce_on_plateau'],
            [float, 'max_lr', 0.1, 'used in one_cycle'],
            [float, 'min_momentum', 0.8, 'used in one_cycle'],
            [float, 'max_momentum', 0.95, 'used in one_cycle'],
            [float, 'phase_1_pct', 0.3, 'used in one_cycle and flat_and_anneal. Percentage of the training dedicated to the first phase'],
            [int, 'max_steps', 200, 'used in lr_finder, onecycle and flat_and_anneal'],
            [int, 'drop_after_num_epoch', 10, 'used in step_decay, reduce lr after specific number of epochs'],
            ['list[int]', 'drop_schedule', [30, 50, 70], 'used in step_decay_schedule and explicit_schedule, reduce lr after specific number of epochs'],
            ['list[float]', 'list_lr', [0.01, 0.001, 0.0001], 'used in explicit_schedule, lr values after specific number of epochs'],
            [float, 'decay_step', 1.0, 'used in inverse time decay, decay_step controls how fast the decay rate reduces '],
            [bool, 'staircase', False, 'if true then return the floor value of inverse_time_decay'],
        ]],
    ]],
]


class GenericScheduler(Callback):
  def __init__(self):
    super(GenericScheduler, self).__init__()

  def get_lr(self):
    return tf.keras.backend.get_value(self.model.optimizer.lr)

  def get_momentum(self):
    return tf.keras.backend.get_value(self.model.optimizer.momentum)

  def set_lr(self, lr):
    tf.keras.backend.set_value(self.model.optimizer.lr, lr)

  def set_momentum(self, mom):
    tf.keras.backend.set_value(self.model.optimizer.momentum, mom)


class LRFinder(GenericScheduler):
  """
  Callback that exponentially adjusts the learning rate after each training batch between start_lr and
  end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
  visually finding a good learning rate.
  """

  def __init__(self, start_lr: float = 1e-6, end_lr: float = 10, max_steps: int = 100, smoothing=0.7, log_dir=None):
    super(LRFinder, self).__init__()
    self.start_lr, self.end_lr = start_lr, end_lr
    self.max_steps = max_steps
    self.smoothing = smoothing
    self.step_lrf, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
    self.lrs, self.losses = [], []
    self.log_dir = log_dir
    self.file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "metrics"))
    self.sanity_check = False

  def on_train_begin(self, logs=None):
    self.step_lrf, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
    self.lrs, self.losses = [], []

  def on_train_batch_begin(self, batch, logs=None):
    self.lr = self.exp_annealing(self.step_lrf)
    self.set_lr(self.lr)

    for m in self.model.metrics:
      m.reset_states()

  def on_train_batch_end(self, batch, logs=None):
    logs = logs or {}
    loss = logs.get('loss')
    step = self.step_lrf
    if loss:
      self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
      smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step_lrf + 1))
      self.losses.append(loss)  # originally smooth loss here
      self.lrs.append(self.lr)
      with self.file_writer.as_default():
        tf.summary.scalar("learning rate by steps", data=self.lr, step=self.step_lrf)
        tf.summary.scalar("loss by steps", data=loss, step=self.step_lrf)
        tf.summary.scalar("smooth loss by steps", data=smooth_loss, step=self.step_lrf)

      if step == 0 or loss < self.best_loss:
        self.best_loss = loss

      if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
        self.model.stop_training = True
        self.plot()

    if step == self.max_steps:
      self.model.stop_training = True
      self.plot()

    self.step_lrf += 1

  def exp_annealing(self, step):
    return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

  def plot(self):
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_xscale('log')
    idx_min = self.losses.index(min(self.losses))

    plot_max_range = max(self.losses[:idx_min])*1.1
    plot_min_range = 0.9 * min(self.losses)
    ax.set_ylim(plot_min_range, plot_max_range)

    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    ax.plot(self.lrs, self.losses)
    fig.savefig(os.path.join(self.log_dir, 'lr_finder.png'))


class CosineAnnealer:

  def __init__(self, start, end, steps):
    self.start = start
    self.end = end
    self.steps = steps
    self.n = 0

  def step(self):
    self.n += 1
    cos = np.cos(np.pi * (self.n / self.steps)) + 1
    return self.end + (self.start - self.end) / 2. * cos


class FlatAndAnnealScheduler(GenericScheduler):
  """
  Callback that follows the flat and anneal learning rate policy.
  The learning rate is constant during the first phase. It then decreases using cosine annealing during the second phase.
  """
  def __init__(self, lr, steps, phase_1_pct=0.8, log_dir=None):
    super(FlatAndAnnealScheduler, self).__init__()
    self.flat_lr = lr
    self.final_lr = lr / 1e6
    phase_1_steps = steps * phase_1_pct

    self.phase_1_steps = phase_1_steps
    self.phase_2_steps = steps - phase_1_steps
    self.step = 0
    self.max_steps = steps

    self.annealer = CosineAnnealer(lr, self.final_lr, self.phase_2_steps)

    self.log_dir = log_dir
    self.file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "metrics"))

  def on_train_batch_begin(self, batch, logs=None):
    lr = self.get_lr()
    with self.file_writer.as_default():
      tf.summary.scalar("learning rate by steps", data=lr, step=self.step)

  def on_train_batch_end(self, batch, logs=None):
    self.step += 1
    if self.step < self.phase_1_steps:
      self.set_lr(self.flat_lr)
    elif self.step > self.max_steps:
      self.model.stop_training = True
    else:
      self.set_lr(self.annealer.step())

  def on_epoch_end(self, epoch, logs=None):
    lr = self.get_lr()
    with self.file_writer.as_default():
      tf.summary.scalar("learning rate by epochs", data=lr, step=epoch)


class OneCycleScheduler(GenericScheduler):
  """
  Callback that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
  If the model supports a momentum parameter, it will also be adapted by the schedule.
  The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
  only two phases are used and the adaptation is done using cosine annealing.
  In phase 1 the LR increases from lrmax÷fac→r to lrmax and momentum decreases from mommax to mommin
.
  In the second phase the LR decreases from lrmax to lrmax÷fac→r⋅1e4 and momemtum from mommax to mommin
.
  By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter phase1_pct
.
  """

  def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25., log_dir=None):
    super(OneCycleScheduler, self).__init__()
    lr_min = lr_max / div_factor
    final_lr = lr_max / (div_factor * 1e4)
    phase_1_steps = steps * phase_1_pct
    phase_2_steps = steps - phase_1_steps

    self.phase_1_steps = phase_1_steps
    self.phase_2_steps = phase_2_steps
    self.phase = 0
    self.step = 0
    self.max_steps = steps

    self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)],
                   [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

    self.lrs = []
    self.moms = []
    self.log_dir = log_dir
    self.file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "metrics"))

  def on_train_batch_begin(self, batch, logs=None):
    logs = logs or {}
    lr = self.get_lr()
    mom = self.get_momentum()
    self.lrs.append(self.get_lr())
    self.moms.append(self.get_momentum())
    with self.file_writer.as_default():
      tf.summary.scalar("learning rate by steps", data=lr, step=self.step)
      tf.summary.scalar("momentum by steps", data=mom, step=self.step)

  def on_train_batch_end(self, batch, logs=None):
    self.step += 1
    if self.step >= self.phase_1_steps:
      self.phase = 1
    if self.step > self.max_steps:
      self.model.stop_training = True

    self.set_lr(self.lr_schedule().step())
    self.set_momentum(self.mom_schedule().step())

  def on_epoch_end(self, epoch, logs=None):
    lr = self.get_lr()
    mom = self.get_momentum()
    with self.file_writer.as_default():
      tf.summary.scalar("learning rate by epochs", data=lr, step=epoch)
      tf.summary.scalar("momentum by epochs", data=mom, step=epoch)

  def lr_schedule(self):
    return self.phases[self.phase][0]

  def mom_schedule(self):
    return self.phases[self.phase][1]


class ExponentialDecay:
  """Applies exponential decay to the learning rate after each epoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (Float): initial learning rate when the tranining starts

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float):
    self.initial_lr = initial_lr

  def __call__(self, epoch: int):
    return tf.maximum(self.initial_lr * tf.pow(1 - 0.1, epoch), _END_LEARNING_RATE)


class StepDecay:
  """Applies Step decay to the learning rate after each epoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the traning starts
      drop_rate (float): drop_rate defines ratio to multiply the learning rate
      drop_after_num_epoch (int): after how many epochs the drop_rate has to be applied

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float, drop_rate=0.5, drop_after_num_epoch=10):
    self.initial_lr = initial_lr
    self.drop_rate = drop_rate
    self.drop_after_num_epoch = drop_after_num_epoch

  def __call__(self, epoch: int):
    return tf.maximum(self.initial_lr * tf.pow(self.drop_rate, tf.floor(epoch / self.drop_after_num_epoch)), _END_LEARNING_RATE)


class StepDecaySchedule:
  """Applies Step decay schedule to the learning rate after each epoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the traning starts
      drop_schedule (list[int]): list of integers to specify which epochs to reduce the lr
      drop_rate (float): drop_rate defines ratio to multiply the learning rate
      total_epochs (int): the length to which the lr decay should be applied

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float, drop_schedule: List[int], drop_rate=0.5, total_epochs=100):
    self.initial_lr = initial_lr
    self.drop_schedule = list(set(drop_schedule + [total_epochs]))
    self.drop_schedule.sort()
    self.drop_rate = drop_rate
    self.total_epochs = total_epochs
    self.built = False

  def build(self):
    assert lambda x: x > 0 in self.drop_schedule
    assert max(self.drop_schedule) <= self.total_epochs
    self.built = True

  def __call__(self, epoch: int):
    if not self.built:
      self.build()

    self.schedule = []
    for i in range(len(self.drop_schedule)):
      # store the learning rate change based on the drop rate.
      self.schedule.append(max(round(self.initial_lr * math.pow(self.drop_rate, i), 5), _END_LEARNING_RATE))
    index = [epoch <= x for x in self.drop_schedule].index(True)  # get count of true values
    return self.schedule[index]  # pick the respective lr rate


class ExplicitSchedule:
  """explicitDecay takes as parameters a list of learning rate and a list of epoch and change the learning rate at theses epochs
  Both list should have the same size

  Args:
    initial_lr (float): initial learning rate when the traning starts
    drop_schedule (list[int]): list of integers to specify which epochs to reduce the lr
    lr_list (list[float]): list of learning rate to apply at each drop_schedule

  """

  def __init__(self, initial_lr: float, drop_schedule: List[int], lr_list: List[float]):
    self.drop_schedule = drop_schedule
    self.lr_list = lr_list
    self.built = False

    # variable so we don't need to explore the list every time
    self.current_lr = initial_lr
    self.current_index = -1

  def build(self):
    assert len(self.drop_schedule) == len(self.lr_list)
    assert len(self.drop_schedule) > 0
    self.built = True

  def __call__(self, epoch: int):
    if not self.built:
      self.build()
    # check if the learning rate need to change
    if self.current_index + 1 != len(self.drop_schedule) and epoch == self.drop_schedule[self.current_index + 1]:
      self.current_index += 1
      self.current_lr = self.lr_list[self.current_index]
    return self.current_lr


class PolynomialDecay:
  """Applies Polynomial decay to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      power (float): nth degree polynomial to be applied]
      total_epochs (int): the lenght to which the decay should be applied

  returns: float adjusted learning rate. 
  """

  def __init__(self, initial_lr: float, power=5.0, total_epochs=100):
    self.initial_lr = initial_lr
    self.power = power
    self.total_epochs = total_epochs

  def __call__(self, epoch: int):
    return ((self.initial_lr - _END_LEARNING_RATE) *
            tf.pow((1 - epoch / self.total_epochs), self.power)
            ) + _END_LEARNING_RATE


class InverseTimeDecay:
  """Applies inverse time decay to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      decay_rate (float): decay_rate to be multiplied at each epoch]
      decay_step (float): this controls the how steep the decay would be applied]
      staircase (bool): applies integer floor division there by producing non continous decay

  returns: float adjusted learning rate. 
  """

  def __init__(self, initial_lr: float, decay_rate=0.5, decay_step=1.0, staircase=False):
    self.initial_lr = initial_lr
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.staircase = staircase

  def __call__(self, epoch: int):
    if self.staircase:
      return self.initial_lr / (1 + tf.floor(self.decay_rate * epoch / self.decay_step))
    else:
      return self.initial_lr / (1 + self.decay_rate * epoch / self.decay_step)


class CosineDecay:
  """Applies cosine decay to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      alpha (float): controls the intensity of the cosine function applied
      total_epochs (int): the lenght to which the decay should be applied

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float, alpha=0.001, total_epochs=100):
    self.initial_lr = initial_lr
    self.alpha = alpha
    self.total_epochs = total_epochs

  def __call__(self, epoch: int):
    cosine = 0.5 * (1 + tf.cos((math.pi * epoch) / self.total_epochs))
    decayed = (1 - self.alpha) * cosine + self.alpha
    return tf.maximum(self.initial_lr * decayed, _END_LEARNING_RATE)


def get_lr_scheduler(lr: float, total_epochs: int, lr_params: dict, log_dir=None):
  lr_schedule_name = lr_params['strategy'].lower()

  get_lr = {
      "exponential_decay": LearningRateScheduler(ExponentialDecay(lr)),
      "step_decay": LearningRateScheduler(StepDecay(
          lr,
          lr_params['decay_rate'],
          lr_params['drop_after_num_epoch'])),
      "step_decay_schedule": LearningRateScheduler(StepDecaySchedule(
          lr,
          lr_params['drop_schedule'],
          lr_params['decay_rate'],
          total_epochs)),
      "explicit_schedule": LearningRateScheduler(ExplicitSchedule(
        lr,
        lr_params['drop_schedule'],
        lr_params['list_lr'],
      )),
      "polynomial_decay": LearningRateScheduler(PolynomialDecay(
          lr,
          lr_params['power'],
          total_epochs)),
      "inverse_time_decay": LearningRateScheduler(InverseTimeDecay(
          lr,
          lr_params['decay_rate'],
          lr_params['decay_step'],
          lr_params['staircase'])),
      "cosine_decay": LearningRateScheduler(CosineDecay(
          lr,
          lr_params['alpha'],
          total_epochs)),
      "lr_reduce_on_plateau": ReduceLROnPlateau(
          monitor='val_loss',
          factor=lr_params['decay_rate'],
          patience=lr_params['patience'],
          verbose=1,
          mode='auto',
          min_lr=lr_params['min_lr']),
      "lr_finder": LRFinder(
          max_steps=lr_params["max_steps"],
          log_dir=log_dir),
      "one_cycle": OneCycleScheduler(
          lr_max=lr_params["max_lr"],
          mom_max=lr_params["max_momentum"],
          mom_min=lr_params["min_momentum"],
          steps=lr_params["max_steps"],
          phase_1_pct=lr_params["phase_1_pct"],
          log_dir=log_dir),
      "flat_and_anneal": FlatAndAnnealScheduler(
          lr=lr,
          phase_1_pct=lr_params["phase_1_pct"],
          steps=lr_params["max_steps"],
          log_dir=log_dir)
  }
  return get_lr[lr_schedule_name]


def get_optimizer(optimizer_param: dict):
  optimizer_name = optimizer_param['name'].lower()
  lr = optimizer_param['lr']

  kwargs = {}
  if optimizer_param['clipnorm'] != 0:
    kwargs['clipnorm'] = optimizer_param['clipnorm']
  if optimizer_param['clipvalue'] != 0:
    kwargs['clipvalue'] = optimizer_param['clipvalue']
  

  optimizer_dict = {
      'adadelta': Adadelta(lr, **kwargs),
      'adagrad': Adagrad(lr, **kwargs),
      'adam': Adam(lr, **kwargs),
      'adam_amsgrad': Adam(lr, amsgrad=True, **kwargs),
      'sgd': SGD(lr, **kwargs),
      'sgd_momentum': SGD(lr, momentum=optimizer_param['momentum'], **kwargs),
      'sgd_nesterov': SGD(lr, momentum=optimizer_param['momentum'], nesterov=True, **kwargs),
      'nadam': Nadam(lr, **kwargs),
      'rmsprop': RMSprop(lr, **kwargs),
      'radam': RectifiedAdam(lr, **kwargs),
  }

  optimizer = optimizer_dict[optimizer_name]

  if optimizer_param['lookahead']:
    optimizer = Lookahead(optimizer=optimizer, sync_period=optimizer_param['sync_period'])


  return optimizer
