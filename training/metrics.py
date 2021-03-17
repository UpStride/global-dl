import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow.python.util import object_identity

def log10(x):
  base = 10.
  return tf.math.log(x) / tf.math.log(base)

def calc_accuracy(y_true, y_pred):
  y_true = tf.math.argmax(tf.convert_to_tensor(y_true, tf.float32), axis=-1)
  y_pred = tf.math.argmax(tf.convert_to_tensor(y_pred, tf.float32), axis=-1)
  return tf.math.reduce_mean(tf.cast((tf.math.equal(y_true, y_pred)), dtype=tf.float32))

def count_trainable_params(model):
  """
  Count the number of trainable parameters of tf.keras model
  Args
      model: tf.keras model
  return
      Total number ot trainable parameters
  """
  weights = model.trainable_weights
  total_trainable_params = int(sum(np.prod(p.shape.as_list()) for p in object_identity.ObjectIdentitySet(weights)))
  return total_trainable_params

def multiply_and_reduce(shape):
  """Function calculates product between the shape (list) of the given layer and outputs a integer
  eg: if shape is [10, 5, 3], the function is going to return ((10*5)*3).

  Args:
      shape (list): list of n values, typically 3

  Returns:
      int 
  """
  return int(reduce(lambda x, y: x*y, shape))

def _count_linear_layer(layer, N):
  """
  Note: This calculates the FLOPs for the unoptimized implementation of any Algebra
  """
  input_shape = layer.input_shape
  output_shape = layer.output_shape
  if len(input_shape) == 4: # 2D Conv and DepthWise
    if layer.data_format == "channels_first":
      input_channels = input_shape[1]
      output_channels, h, w, = output_shape[1:]
    elif layer.data_format == "channels_last":
      input_channels = input_shape[3]
      h, w, output_channels = output_shape[1:]
    w_h, w_w = layer.kernel_size
    class_name = layer.__class__.__name__
    if class_name == "DepthwiseConv2D": 
      output_channels = 1
    # grouping the products to input and outputs
    input_group = w_h * w_w * input_channels
    output_group = h * w * output_channels
  elif len(input_shape) == 2: # Dense 
    # grouping the products to input and outputs
    input_group = input_shape[1] 
    output_group = output_shape[1]
  else:
    raise NotImplementedError("Flops for {layer.name} layer not implemented")

  n_mul = (N**2) * (input_group * output_group)
  n_add = (N*(N-1)) * (input_group * output_group)
  
  flops = n_mul + n_add

  if N == 1: 
    flops *= 2 # n_add becomes zero for N = 1, hence multiplying to get the correct FLOPs

  if layer.use_bias:
    flops += (output_group * N)

  return int(flops)

def _count_flops_relu(layer, N):
  """ Dev note : current tensorflow profiler say ReLU doesn't cost anything...
  """
  # 2 operations per component : compare and assign
  return N * (multiply_and_reduce(layer.output_shape[1:]) * 2)

def _count_flops_hard_sigmoid(layer, N):
  relu = _count_flops_relu(layer, N=1)
  add_divide = multiply_and_reduce(layer.output_shape[1:]) * 2
  # relu + one addtion and one division
  count = relu + add_divide
  return N * count 

def _count_flops_hard_swish(layer, N):
  hard_sigmoid = _count_flops_hard_sigmoid(layer, N=1)
  x = multiply_and_reduce(layer.output_shape[1:])
  return N * (hard_sigmoid + x) # hard_sigmoid + 1 multiplication

def _count_flops_maxpool2d(layer, N):
  return N * (layer.pool_size[0] * layer.pool_size[1] * multiply_and_reduce(layer.output_shape[1:]))

def _count_flops_global_avg_max_pooling(layer, N):
  """
  This function can be used the count FLOPs for the below layers 
  GlobalAveragePool2D
  GlobalMaxpool2D
  """
  return N * multiply_and_reduce(layer.input_shape[1:])

def _count_flops_add_mul(layer, N):
  """
  This function can be used the count FLOPs for the below layers 
  Add
  Multiply
  """
  return N * multiply_and_reduce(layer.output_shape[1:])

def format_flops(flops):
  if flops // 10e9 > 0:
    return str(round(flops / 10.e9, 2)) + ' GFLOPs'
  elif flops // 10e6 > 0:
    return str(round(flops / 10.e6, 2)) + ' MFLOPs'
  elif flops // 10e3 > 0:
    return str(round(flops / 10.e3, 2)) + ' KFLOPs'
  else:
    return str(round(flops), 2) + ' FLOPs'

def get_map_types(upstride_type):
  map_type = {
    -1: 1,
     0: 1,
     1: 2,
     2: 4,
     3: 8,
  }
  return map_type[upstride_type]

def count_flops_efficient(model, upstride_type=-1):
  N = get_map_types(upstride_type) 

  flops = 0

  # Not all the activations are present in keras layers. 
  # TODO add new layers to the engine for both tensorflow and upstride.
  map_activation = {
    "relu": _count_flops_relu,
    "hard_sigmoid": _count_flops_hard_sigmoid,
    "hard_swish": _count_flops_hard_swish,
    "softmax": lambda x,y: 0 # TODO plan to skip 
  }

  map_layer_to_count_fn = {
    "Conv2D": _count_linear_layer,
    "DepthwiseConv2D": _count_linear_layer,
    "Dense": _count_linear_layer,
    "ReLU": _count_flops_relu,
    "MaxPooling2D": _count_flops_maxpool2d,
    "GlobalMaxPooling2D": _count_flops_global_avg_max_pooling,
    "GlobalAveragePooling2D": _count_flops_global_avg_max_pooling,
    "Add": _count_flops_add_mul,
    "Multiply": _count_flops_add_mul
  }

  for i, layer in enumerate(model.layers):
    layer_class_name = layer.__class__.__name__
    if layer_class_name in map_layer_to_count_fn:
      # print(i, layer)
      flops += map_layer_to_count_fn[layer_class_name](layer, N) 
    if type(layer) == tf.keras.layers.Activation:
      flops += map_activation[layer.activation.__name__](layer, N)
        
  # return format_flops(int(flops))
  return int(flops)

def count_flops(model):
  """
  Count the number of FLOPS of tf.keras model
  Args
      model: tf.keras model
  return
      Total number of FLOPS
  """
  session = tf.compat.v1.Session()
  graph = tf.Graph()
  with graph.as_default():
    with session.as_default():
      # Make temporary clone of our model under the graph
      temp_model = tf.keras.models.clone_model(model)
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
  # To avoid flops accumulation for multiple run, reset the graph
  del graph
  return flops.total_float_ops


def information_density(model):
  """
  Calculate accuracy per M params introduced in this paper (https://arxiv.org/pdf/1605.07678.pdf)
  """
  def metric(y_true, y_pred):
    # Counting parameters in millions
    total_params = count_trainable_params(model) / 1.0e6
    accuracy = calc_accuracy(y_true, y_pred) * 100.0
    info_density = accuracy / total_params
    return info_density
  return metric

def net_score(model, alpha=2.0, beta=0.5, gamma=0.5):
  """
  Calculate custom evaluation metrics for energy efficient model by considering accuracy, computational cost and
  memory footprint, introduced in this paper (https://arxiv.org/pdf/1806.05512.pdf)
  Args
      model: tf keras model
      alpha: coefficient that controls the influence of accuracy
      beta:  coefficient that controls the influence of architectural complexity
      gamma: coefficient that controls the influence of computational complexity

  """
  def metric(y_true, y_pred):
    # Counting parameters in millions
    total_params = count_trainable_params(model) / 1.0e6
    # Counting MACs in Billions (assuming 1 MAC = 2 FLOPS)
    total_MACs = ((count_flops(model) / 2.0) / 1.0e9)
    accuracy = calc_accuracy(y_true, y_pred) * 100.0
    score = 20 * log10(tf.math.pow(accuracy, alpha) / (tf.math.pow(total_params, beta) * tf.math.pow(total_MACs, gamma)))
    return score
  return metric

# custom metrices  by extending tf.keras.metrics.Metric
class InformationDensity(tf.keras.metrics.Metric):
  """
  Calculate accuracy per M params introduced in this paper (https://arxiv.org/pdf/1605.07678.pdf)

  """

  def __init__(self, model, name='information_density', **kwargs):
    super(InformationDensity, self).__init__(name=name, **kwargs)
    self.model = model
    self.info_density = self.add_weight(name='info_density', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):

    info_density = information_density(self.model)(y_true, y_pred)

    self.info_density.assign_add(info_density)

  def result(self):
    return self.info_density

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.info_density.assign(0.)


class NetScore(tf.keras.metrics.Metric):
  """
      Calculate custom evaluation metrics for energy efficient model by considering accuracy, computational cost and
      memory footprint, introduced in this paper (https://arxiv.org/pdf/1806.05512.pdf)
      Args
          model: tf keras model
          alpha: coefficient that controls the influence of accuracy
          beta:  coefficient that controls the influence of architectural complexity
          gamma: coefficient that controls the influence of computational complexity

      """

  def __init__(self, model, alpha=2.0, beta=0.5, gamma=0.5, name='net_score', **kwargs):
    super(NetScore, self).__init__(name=name, **kwargs)
    self.model = model
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.net_score = self.add_weight(name='netscore', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):

    score = net_score(self.model)(y_true, y_pred)

    self.net_score.assign_add(score)

  def result(self):
    return self.net_score

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.net_score.assign(0.)
