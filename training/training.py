import os
from typing import Dict
import tensorflow as tf
from .optimizers import get_lr_scheduler


def create_dir(path: str):
  """this function exists to be called by the argument parser,
  to automatically create new directories
  """
  try:
    os.makedirs(path, exist_ok=True)
  except FileExistsError as e:
    # this error shouldn't happen because of exist_ok=True, but we never know
    return False
  except FileNotFoundError as e:
    return False
  return True


def create_dir_or_empty(path: str):
  """this function exists to be called by the argument parser,
  to automatically create new directories if path is not empty
  """
  if path == "":
    return True
  return create_dir(path)


# list of [type, name, default, help, condition] or ['namespace', name, List]
# if condition is specified and false, then raise an exception
# type can be :
#  - one of the following python types : int, str, bool, float
#  - 'list[{type}]' with type in [int, str, bool, float] (for instance 'list(str)')
#  - 'namespace' to define namespace
arguments = [
    ['list[str]', 'yaml_config', [], 'config file overriden by these argparser parameters'],
    [str, 'checkpoint_dir', '', 'checkpoints directory', create_dir],
    [int, 'max_checkpoints', 5, 'maximum number of checkpoints to keep (from the last epoch)'],
    [int, 'checkpoint_freq', 1, 'frequency of saving the checkpoints; e.g. checkpoint_freq=5 saves checkpoints every 5 epochs'],
    [str, 'title', '', 'title of the experiment'],
    [str, 'description', '', 'description of the experiment'],
    [str, 'log_dir', '', 'Log directory', create_dir],
    [int, 'num_epochs', 60, 'The number of epochs to run', lambda x: x > 0],
    [int, 'early_stopping', 1000000, 'stop  the training if validation loss doesn\'t decrease for n value'],
    ['namespace', 'debug', [
        [bool, 'write_graph', False, ''],
        [bool, 'write_histogram', False, ''],
        [bool, 'log_gradients', False, 'whether to visualize the histogram, distribution and norm of gradients'],
        [bool, 'profiler', False, 'if true then profile tensorflow training using tensorboard. Need tf >=2.2'],
    ]]
]


def create_env_directories(experiment_name: str, checkpoint_dir: str, log_dir: str, export_dir=''):
  """ Create the checkpoint, log and export directories by joining the experiment_name to the provided directories.

  Returns: The updated directories. export_dir is None if the user don't want to export the result of the training
  """
  checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
  log_dir = os.path.join(log_dir, experiment_name)
  export_dir = os.path.join(export_dir, experiment_name) if export_dir else None
  return checkpoint_dir, log_dir, export_dir


class GradientCallback(tf.keras.callbacks.Callback):
  """ Custom callback for gradient visualization in tensorboard
  """

  def __init__(self, batch, log_dir, log_freq=0):
    """ 
    Args:
      log_dir (str): the path of the directory where to save the log files to be parsed by TensorBoard.
      batch (tuple): batch of image and label pair, gradient will be calculated on this
      log_freq (int): frequency (in epochs) at which to visualize the gradients. If set to 0, gradients won't be vizualized
    """
    self.log_freq = log_freq
    self.batch = batch
    self.grad_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'grads'))

  def _log_gradients(self, epoch):
    """Logs the gradients of the Model."""
    x, y = self.batch
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    trainable_vars = self.model.trainable_variables
    with tf.GradientTape() as tape:
      y_pred = self.model(x, training=False)
      loss = self.model.compiled_loss(y, y_pred, regularization_losses=self.model.losses)
    gradients = tape.gradient(loss, trainable_vars)
    with self.grad_writer.as_default():
      for weights, grads in zip(trainable_vars, gradients):
        # keeping only the grads for kernels and biases
        if 'kernel' in weights.name or 'bias' in weights.name:
          grad_name = weights.name.replace(':', '_')
          if 'kernel' in weights.name:
            grad_name = 'kernel_grad/' + grad_name
          if 'bias' in weights.name:
            grad_name = 'bias_grad/' + grad_name
          g_norm = tf.norm(grads, ord='euclidean')
          tf.summary.histogram(grad_name+'_hist', grads, epoch)
          tf.summary.scalar(grad_name+'_norm', g_norm, epoch)

  def on_epoch_end(self, epoch, logs=None):
    if self.log_freq and epoch % self.log_freq == 0:
      self._log_gradients(epoch)


def get_callbacks(config, log_dir):
  # define callbacks
  histogram_freq = 1 if config['debug']['write_histogram'] else 0
  write_graph = config['debug']['write_graph']
  profile_batch = '10, 12' if config['debug']['profiler'] else 0
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, write_graph=write_graph, write_images=False, profile_batch=profile_batch)
  callbacks = [tensorboard_cb, tf.keras.callbacks.EarlyStopping('val_loss', patience=config['early_stopping'])]
  if config['optimizer']['lr_decay_strategy']['activate']:
    callbacks.append(
        get_lr_scheduler(config['optimizer']['lr'], config['num_epochs'], config['optimizer']['lr_decay_strategy']['lr_params'], log_dir)
    )
  return callbacks


def init_custom_checkpoint_callbacks(trackable_objects: Dict, ckpt_dir: str, max_ckpt: int, save_frequency: int):
  """ Prepare a checkpoint callbacks. this function also load back the last checkpoint and return the current epoch number

  Args:
      trackable_objects: dictionary mapping a tag to a tensorflow object to track. Can be models or optimizers
      ckpt_dir: path where to write the checkpoint
      max_ckpt: maximum number of checkpoint to keep
      save_frequency: frequence to save the checkpoint

  Returns: the tensorflow Callback anf the last epoch
  """
  checkpoint = tf.train.Checkpoint(**trackable_objects)
  manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir, max_to_keep=max_ckpt)
  latest = manager.restore_or_initialize()
  latest_epoch = 0
  if latest is not None:
    print(f'restore {manager.latest_checkpoint}')
    latest_epoch = int(manager.latest_checkpoint.split('-')[-1])
  return tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: manager.save(checkpoint_number=epoch) if (epoch % save_frequency) == 0 else None), latest_epoch
