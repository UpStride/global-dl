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
  total_trainable_params = int(sum(np.prod(p.shape) for p in object_identity.ObjectIdentitySet(weights)))
  return total_trainable_params


def _count_linear_layer(layer, num_of_blades):
  """ This function calculates the flops for any linear layer
  Algorithm Explaination for Real Number: 
  - The Number of operation in any linear layer is the number of the operation to compute one element in the output tensor times the size of the output tensor
  - The size of the output tensor can be find by the looking at the output of the operation.
  - The number of multiplcation to compute one element is the size of the kernel and number of additions is size of the kernel -1. 

  Step 1: Find the size of the output Tensor
  Step 2: Find the size of the kernel 
  Step 3: Compute the number of operations. 

  Args:
    layer (tf.keras.layer.Conv2D or DepthwiseConv2D or Dense): any linear layer from the keras model
    num_of_blades: This value represent the number of blades we use for different upstride types.

  Returns:
      int : total FLOPs for the given layer.
  """
  input_shape = layer.input_shape
  output_shape = layer.output_shape
  class_name = layer.__class__.__name__

  # Step 1: 
  n_elements_in_output_tensor = np.prod(output_shape[1:])

  # Step 2: 
  # we get kernel size (this is layer dependant (Dense vs Conv))
  if class_name == "Conv2D":
    channel_index = 1 if layer.data_format == "channels_first" else -1
    input_channels = input_shape[channel_index]
    n_single_mult = np.prod(layer.kernel_size) * input_channels
    # Implementation Notes: This formulation works for group convolutions because the input channels = c_in / group, where c_in number of channel in the input tensor
  elif class_name == "DepthwiseConv2D":
    n_single_mult = np.prod(layer.kernel_size)
  elif class_name == "Dense":
    n_single_mult = input_shape[1]
  else:
    raise NotImplementedError("FLOPs for {layer.name} layer not implemented")

  # Step 3:
  n_mult_real = n_single_mult * n_elements_in_output_tensor
  n_mul = (num_of_blades**2) * n_mult_real
  # max(1, num_of_blades-1) to ensure the n_add doesn't becomes zero when num_of_blades = 1
  n_add = num_of_blades * (max(1, num_of_blades-1)) * n_mult_real
  
  flops = n_mul + n_add

  if layer.use_bias:
    flops += n_elements_in_output_tensor * num_of_blades

  return int(flops)


def _count_flops_relu(layer):
  """ Dev note : current tensorflow profiler say ReLU doesn't cost anything...
  """
  # 2 operations per component : compare and assign
  return np.prod(layer.output_shape[1:]) * 2


def _count_flops_hard_sigmoid(layer):
  """count FLOPs for hard_sigmoid activation

  Args:
      layer (tf.keras.layers.Activation): Activation layer

  Returns:
      int: FLOPs
  """
  relu = _count_flops_relu(layer)
  add_divide = np.prod(layer.output_shape[1:]) * 2
  # relu + one addition and one division
  return int(relu + add_divide)


def _count_flops_hard_swish(layer):
  """count FLOPs for hard_swish activation

  Args:
      layer (tf.keras.layers.Activation): Activation layer

  Returns:
      int: FLOPs
  """
  hard_sigmoid = _count_flops_hard_sigmoid(layer)
  x = np.prod(layer.output_shape[1:])
  return int(hard_sigmoid + x) # hard_sigmoid + 1 multiplication


def _count_flops_maxpool_avgpool(layer):
  """count flops for layer max pool 2d and average pooling 2d

  args:
      layer (tf.keras.layers.MaxPooling2D or AveragePooling2D): maxpooling or averagepooling layer

  returns:
      int: flops
  """
  return layer.pool_size[0] * layer.pool_size[1] * np.prod(layer.output_shape[1:])


def _count_flops_global_avg_max_pooling(layer):
  """count flops for layer global (MaxPooling2D or AvgPooling2D)

  args:
      layer (tf.keras.layers.GlobalMaxPooling2D or tf.keras.layers.GlobalAveragPooling2D)

  returns:
      int: flops
 
  This function can be used the count FLOPs for the below layers 
  GlobalAveragePooling2D
  GlobalMaxPooling2D
  """
  return np.prod(layer.input_shape[1:])


def _count_flops_add_mul(layer):
  """count flops for layer Add or Multiply

  args:
      layer (tf.keras.layers.Add or tf.keras.layers.Multiply)

  returns:
      int: flops

  This function can be used the count FLOPs for the below layers 
  Add
  Multiply
  """
  return np.prod(layer.output_shape[1:])


def format_flops(flops):
  """Formats the FLOPs into specific category depending on how large the value is

  Args:
      flops (int): Over all FLOPs for the model

  Returns:
      [str]: FLOPs 2 decimal places
  """
  if flops // 1e9 > 0:
    return str(round(flops / 1e9, 2)) + ' GFLOPs'
  elif flops // 1e6 > 0:
    return str(round(flops / 1e6, 2)) + ' MFLOPs'
  elif flops // 1e3 > 0:
    return str(round(flops / 1e3, 2)) + ' KFLOPs'
  else:
    return str(round(flops, 2)) + ' FLOPs'


def get_map_types(upstride_type):
  """
  maps the upstride_type to dictionary map_type and returns the value
  """
  map_type = {
    -1: 1,
     0: 1,
     1: 2,
     2: 4,
     3: 8,
  }
  return map_type[upstride_type]


def get_map_activations(activation_name):
  """
  maps the activation name to dictionary map_activation and returns the value
  Not all the activations are present in keras layers. 
  Note user should have defined a function with same key name from the below dictionary
  """
  map_activation = {
    "relu": _count_flops_relu,
    "hard_sigmoid": _count_flops_hard_sigmoid,
    "hard_swish": _count_flops_hard_swish,
    "softmax": lambda x: 0 # To ensure class Activation doesn't get report in layers_not_counted_for_flops_calc when activation function is softmax
  }
  return map_activation.get(activation_name)


def get_linear_layer_count_function(layer_class_name):
  """
  maps the layer class name to dictionary map_linear_layer_to_count_fn and returns the value
  if key not found returns None
  """
  map_linear_layer_to_count_fn = {
    "Conv2D": _count_linear_layer,
    "Dense": _count_linear_layer,
    "DepthwiseConv2D": _count_linear_layer
  }
  return map_linear_layer_to_count_fn.get(layer_class_name)


def get_non_linear_count_function(layer_class_name):
  """
  maps the layer class name to dictionary map_non_linear_layer_to_count_fn and returns the value
  if key not found returns None
  """
  map_non_linear_layer_to_count_fn = {
    "ReLU": _count_flops_relu,
    "MaxPooling2D": _count_flops_maxpool_avgpool,
    "AveragePooling2D": _count_flops_maxpool_avgpool,
    "GlobalMaxPooling2D": _count_flops_global_avg_max_pooling,
    "GlobalAveragePooling2D": _count_flops_global_avg_max_pooling,
    "Add": _count_flops_add_mul,
    "Multiply": _count_flops_add_mul
  }
  return map_non_linear_layer_to_count_fn.get(layer_class_name)

def get_layers_not_included_in_flop_count(layers_set):
  """compare layers that currently are not included in the FLOPs count
  If any layer that is not part of the list layers_not_included_in_flop_count will be returned
  If values in layers_set and layers_not_included_in_flop_count are matching empty list will be returned

  Dev note: If you don't want a layer to be reported in the flop count it must be added to the below list

  Args:
      layers_set (set): set of keras layer class name or activation name

  Returns: 
      (list): Delta values between the 2 list or Empty list if values are matching
  """
  layers_not_included_in_flop_count = [
    "InputLayer",
    "BatchNormalization",
    "Flatten",
    "Concatenate",
    "ZeroPadding2D",
    "Reshape",
    "Dropout",
    "Upstride2TF",
    "TF2Upstride",
    "Lambda"
  ]
  return [name for name in layers_set if name not in layers_not_included_in_flop_count]

def count_flops_efficient(model, upstride_type=-1):
  """computes the FLOPs for each layer in the given model and return the total FLOPs for the same

  Args:
      model (tf.keras.Model): Model 
      upstride_type (int, optional): Used to get the number of blades. Defaults to -1.

  Returns:
      [str]: FLOPs for the given Model
  """
  # get the number of blades
  num_of_blades = get_map_types(upstride_type) 
  # initialize flops to 0
  flops = 0

  layers_with_no_flop_count = list()

  for i, layer in enumerate(model.layers):
    layer_class_name = layer.__class__.__name__
    # linear layer
    if get_linear_layer_count_function(layer_class_name) is not None:
      flops += get_linear_layer_count_function(layer_class_name)(layer, num_of_blades) 
    # non linear layer
    elif get_non_linear_count_function(layer_class_name) is not None:
      flops += get_non_linear_count_function(layer_class_name)(layer) * num_of_blades 
    # Activation
    elif isinstance(layer, tf.keras.layers.Activation):
      layer_activation_name = layer.activation.__name__
      if get_map_activations(layer_activation_name) is not None:
        flops += get_map_activations(layer_activation_name)(layer) * num_of_blades
      else:
        layers_with_no_flop_count.append(layer_activation_name)
    else:
      layers_with_no_flop_count.append(layer_class_name)
  
  # get list of layers that are not included in the flop 
  list_layers_with_no_flop = get_layers_not_included_in_flop_count(set(layers_with_no_flop_count))
  # if list not empty print the missing layers
  if list_layers_with_no_flop:
    print(f"FLOPs not calculated for layer(s) {list_layers_with_no_flop}")
  return format_flops(int(flops))


def count_flops(model):
  """
  Count the number of FLOPS of tf.keras model
  Args
      model: tf.keras.Model
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
