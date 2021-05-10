import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

STRATEGIES = [
    "oneDevice",
    "mirrored"
]

arguments = [
    [bool, "xla", False, "In some cases, using xla can speed up training or inference"],
    [bool, "full_gpu_memory", False, "By default, the model will take only what it needs as GPU memory. By turning on this option, it will use the whole GPU memory"],
    [bool, "mixed_precision", False, 'To train with mixed precision'],
    # to use "mirrored" strategy with docker make sure you have enough shared memory
    # you can set the shared memory with the --shm-size option of docker run
    [str, "strategy", "oneDevice", 'tensorflow distribute strategy, can be oneDevice or mirrored', lambda x: x in STRATEGIES],
]


def config_tf2(config):
  """ By default tensorflow 2 take the whole memory of the GPU. For shared server, we may want to change this configuration using "set_memory_growth".

  Args:
      config (dict): dictionary containing 'xla' and 'full_gpu_memory'
  """
  if config['xla']:
    tf.config.optimizer.set_jit(True)
  if not config['full_gpu_memory']:
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
      tf.config.experimental.set_memory_growth(physical_device, True)


def setup_mp(config):
  if config['mixed_precision']:
    print('Training with Mixed Precision')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    print(f'Loss scale: {policy.loss_scale}')
    # the LossScaleOptimizer is not needed because model.fit already handle this. See https://www.tensorflow.org/guide/keras/mixed_precision
    # for more information. I let the code here to remember if one day we go to custom training loop
    # opt = mixed_precision.LossScaleOptimizer(opt, loss_scale=policy.loss_scale)


def setup_strategy(strategy: str):
  """create a tensorflow distribution strategy and return it

  Args:
      strategy (str): name of the strategy. should be in STRATEGIES

  Return: the tensorflow strategy object
  """

  if strategy == STRATEGIES[1]:
    ds_strategy = tf.distribute.MirroredStrategy()
    print('All devices: %s', tf.config.list_physical_devices('GPU'))
  elif tf.config.list_physical_devices('GPU'):
    ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
  else:
    ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  return ds_strategy
