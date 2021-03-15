import numpy as np
import tensorflow as tf
from functools import reduce, partial
from tensorflow.python.util import object_identity
from upstride.type2.tf.keras import layers as type2_layers

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

def _linear_layer(layer, N):
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
    if layer.name.lower() == "depthwise_conv2d": 
      output_channels = 1
  elif len(input_shape) == 2: # Dense 
    input_channels = input_shape[1] 
    output_channels = output_shape[1]
    w_h, w_w, h, w = 1, 1, 1, 1 # setting this to 1
  else:
    raise NotImplementedError("Flops for {layer.name} layer not implemented")

  n_mul = (N**2) * (w_h * w_w * input_channels * output_channels * h * w)
  n_add = (N*(N-1)) * (w_h * w_w * input_channels * output_channels * h * w)
  
  flops = n_mul + n_add

  if N == 1: 
    flops *= 2 # n_add becomes zero for N = 1

  if layer.use_bias:
    flops += output_channels * h * w * N

  return int(flops)

def _count_flops_relu(layer, N):
  """ Dev note : current tensorflow profiler say ReLU doesn't cost anything...
  """
  # 2 operations per component : compare and assign
  return N * (reduce(lambda x, y: x*y, layer.output_shape[1:]) * 2)

def _count_flops_hard_sigmoid(layer, N):
  return N * (_count_flops_relu(layer, N=1) * 2) # relu + one addtion and one division

def _count_flops_hard_swish(layer, N):
  return N * (_count_flops_hard_sigmoid(layer, N=1) * 2) # hard_sigmoid + 1 multiplication

def _count_flops_maxpool2d(layer, N):
  return N * (layer.pool_size[0] * layer.pool_size[1] * reduce(lambda x, y: x*y, layer.output_shape[1:]))

def _count_flops_global_avg_max_pooling(layer, N):
  """
  This function can be used the count FLOPs for the below layers 
  GlobalAveragePool2D
  GlobalMaxpool2D
  """
  return N * (reduce(lambda x, y: x*y, layer.input_shape[1:]))

def _count_flops_add_mul(layer, N):
  """
  This function can be used the count FLOPs for the below layers 
  Add
  Multiply
  """
  return N * (reduce(lambda x, y: x*y, layer.output_shape[1:]))

def _count_flops_dense(layer, N):
  n_mult = layer.input_shape[1] * layer.output_shape[1]
  n_add = layer.input_shape[1] * layer.output_shape[1]
  flops = n_mult + n_add
  if layer.use_bias:
    flops += layer.output_shape[1]
  return int(flops)

def _count_flops_depthwiseconv2d(layer, N):
  if layer.data_format == "channels_first":
    output_channels, h, w, = layer.output_shape[1:]
  elif layer.data_format == "channels_last":
    h, w, output_channels = layer.output_shape[1:]
  w_h, w_w = layer.kernel_size

  n_neurons_output = h * w * output_channels
  n_mult = w_h * w_w * n_neurons_output
  n_add = (w_h * w_w) * n_neurons_output

  flops = n_mult + n_add

  if layer.use_bias:
    flops += n_neurons_output

  return int(flops)

def format_flops(flops):
  if flops // 10e9 > 0:
    return str(round(flops / 10.e9, 2)) + ' GFLOPs'
  elif flops // 10e6 > 0:
    return str(round(flops / 10.e6, 2)) + ' MFLOPs'
  elif flops // 10e3 > 0:
    return str(round(flops / 10.e3, 2)) + ' KFLOPs'
  else:
    return str(round(flops), 2) + ' FLOPs'

def get_type(up_type):
  """Function import specific upstride module depending on the type and returns the corresponding value
  of N which used in FLOP calculation depending on the upstride module

  Args:
      up_type (int): A integer value is passed ranging from -1 till 3.
      Note: this value can go up depending on new upstride types that are introduced.

  Returns:
      upstirde layer, Int: Return the specific upstride import module and the respective N value
  """
  if up_type == -1:
    return tf.keras.layers, 1
  if up_type == 0:
    import upstride.type0.tf.keras.layers as up_layers
    return up_layers, 1
  if up_type == 1:
    import upstride.type1.tf.keras.layers as up_layers
    return up_layers, 2
  if up_type == 2:
    import upstride.type2.tf.keras.layers as up_layers
    return up_layers, 4
  if up_type == 3:
    import upstride.type3.tf.keras.layers as up_layers
    return up_layers, 8

def count_flops_efficient(model, upstride_type=-1):
  layers, N = get_type(upstride_type) 

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
      layers.Conv2D: _linear_layer,
      layers.DepthwiseConv2D: _linear_layer,
      layers.Dense: _linear_layer,
      layers.ReLU: _count_flops_relu,
      layers.MaxPooling2D: _count_flops_maxpool2d,
      layers.GlobalMaxPooling2D: _count_flops_global_avg_max_pooling,
      layers.GlobalAveragePooling2D: _count_flops_global_avg_max_pooling,
      layers.Add: _count_flops_add_mul,
      layers.Multiply: _count_flops_add_mul
  }

  for i, layer in enumerate(model.layers):
    if type(layer) in map_layer_to_count_fn:
      # print(i, layer)
      flops += map_layer_to_count_fn[type(layer)](layer, N) 
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
