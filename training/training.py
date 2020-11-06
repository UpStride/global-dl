import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from .optimizers import get_lr_scheduler, learning_schedule_list


def create_dir(path: str):
  """this function exists to be called by the argument parser,
  to automaticaly create new directories
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
  to automaticaly create new directories if path is not empty
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
    ['list[str]', "yaml_config", [], "config file overriden by these argparser parameters"],
    [str, "checkpoint_dir", '', 'Checkpoints directory', create_dir],
    [str, 'title', '', 'title of the experiment'],
    [str, 'description', '', 'description of the experiment'],
    [str, "log_dir", '', 'Log directory', create_dir],
    [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],  # TODO this number should be computed from dataset
    [int, "num_epochs", 60, 'The number of epochs to run', lambda x: x > 0],
    ['namespace', 'configuration', [
        [bool, "mirrored", False, 'If true then use mirrored strategy'],
        [bool, "with_mixed_precision", False, 'To train with mixed precision'],
        [bool, 'profiler', False, 'if true then profile tensorflow training using tensorboard. Need tf >=2.2'], # TODO move to debug namespace
    ]],
    ['list[int]', "input_size", [224, 224, 3], 'processed shape of each image'],
    [int, 'early_stopping', 20, 'stop  the training if validation loss doesn\'t decrease for n value'],
    ['namespace', 'debug', [
        [bool, 'write_graph', False, ''],
        [bool, 'write_histogram', False, ''],
        [bool, 'log_gradients', False, 'whether to visualize the histogram, distibution and norm of gradients'],
    ]]
]


def create_env_directories(args, experiment_name):
  """
  Args:
      args ([type]): dict need to have checkpoint_dir, log_dir, export_dir
      experiment_name ([type]): [description]

  Returns:
      [type]: [description]
  """
  checkpoint_dir = os.path.join(args['checkpoint_dir'], experiment_name)
  log_dir = os.path.join(args['log_dir'], experiment_name)
  export_dir = os.path.join(args['export']['dir'], experiment_name) if args['export']['dir'] else None
  return checkpoint_dir, log_dir, export_dir


def setup_mp(args):
  if args['configuration']['with_mixed_precision']:
    print('Training with Mixed Precision')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    print(f'Loss scale: {policy.loss_scale}')
    # the LossScaleOptimizer is not needed because model.fit already handle this. See https://www.tensorflow.org/guide/keras/mixed_precision
    # for more information. I let the code here to remember if one day we go to custom training loop
    # opt = mixed_precision.LossScaleOptimizer(opt, loss_scale=policy.loss_scale)


def define_model_in_strategy(args, get_model):
  if args['configuration']['mirrored']:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model, optimizer = get_model(args)
  else:
    model, optimizer = get_model(args)
  return model, optimizer

class GradientCallback(tf.keras.callbacks.Callback):
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


def get_callbacks(args, log_dir):
  # define callbacks
  histogram_freq = 1 if args['debug']['write_histogram'] else 0
  write_graph = args['debug']['write_graph']
  profile_batch = '10, 12' if args['configuration']['profiler'] else 0
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, write_graph=write_graph, write_images=False, profile_batch=profile_batch)
  callbacks = [tensorboard_cb, tf.keras.callbacks.EarlyStopping('val_loss', patience=args['early_stopping'])]
  if args['optimizer']['lr_decay_strategy']['activate']:
    callbacks.append(
        get_lr_scheduler(args['optimizer']['lr'], args['num_epochs'], args['optimizer']['lr_decay_strategy']['lr_params'])
    )
  return callbacks


def init_custom_checkpoint_callbacks(trackable_objects, ckpt_dir):
  checkpoint = tf.train.Checkpoint(**trackable_objects)
  manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir, max_to_keep=5)
  latest = manager.restore_or_initialize()
  latest_epoch = 0
  if latest is not None:
    print(f'restore {manager.latest_checkpoint}')
    latest_epoch = int(manager.latest_checkpoint.split('-')[-1])
  return tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: manager.save()), latest_epoch
