import unittest
import numpy as np
import tensorflow as tf
from training.metrics import (count_flops, count_trainable_params, InformationDensity, 
                              NetScore, count_flops_efficient, format_flops)

class TestMetrics(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None,
                                                      input_tensor=tf.keras.Input(shape=(224, 224, 3)))
    cls.y_true = [0, 1, 2, 3]
    cls.y_pred = [3, 1, 2, 3]

    cls.total_params = count_trainable_params(cls.model) / 1.0e6
    cls.total_macs = ((count_flops(cls.model) / 2.0) / 1.0e9)

  def test_information_density(self):
    acc = np.mean(np.array(self.y_true) == np.array(self.y_pred)) * 100.0

    true_info_density = acc / self.total_params

    calculated_info_density = InformationDensity(self.model)(tf.one_hot(self.y_true, 4), tf.one_hot(self.y_pred, 4))

    self.assertAlmostEqual(true_info_density, calculated_info_density.numpy(), places=3)

  def test_net_score(self):
    alpha = 2.0
    beta = 0.5
    gamma = 0.5
    acc = np.mean(np.array(self.y_true) == np.array(self.y_pred)) * 100.0

    true_net_score = 20 * np.log10(np.power(acc, alpha) / (np.power(self.total_params, beta) * np.power(self.total_macs, gamma)))

    calculated_net_score = NetScore(self.model)(tf.one_hot(self.y_true, 4), tf.one_hot(self.y_pred, 4))

    print(calculated_net_score)

    self.assertAlmostEqual(true_net_score, calculated_net_score.numpy(), places=3)

class TestCountFlops(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.map_type = {
      -1: 1,
       0: 1,
       1: 2,
       2: 4,
       3: 8,
    }
    cls.input_shape = (100, 50, 10)
    cls.kernel = (5, 3)
    cls.filters = 20

  def relu(self, x):
    return tf.nn.relu(x)

  def hard_sigmoid(self, x):
    return tf.nn.relu6(x + 3.) / 6.

  def hard_swish(self, x):
    return x * self.hard_sigmoid(x)

  def generic_test(self, input, output, upstride_type):
    model = tf.keras.Model(input, output) 
    efficient_count = count_flops_efficient(model, upstride_type)
    return efficient_count
      
  def test_conv2d_without_bias(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Conv2D(self.filters, self.kernel, use_bias=False)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # (5 * 3 * 10 * 20 * 96 * 48) * 2
    self.assertEqual(ef, format_flops(27648000)) 

    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(27648000)) 

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
    # (2**2 * (5 * 3 * 10 * 20 * 96 * 48)) + (2 * (2-1)* (5 * 3 * 10 * 20 * 96 * 48))
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(82944000))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # (4**2 * (5 * 3 * 10 * 20 * 96 * 48)) + (4 * (4-1)* (5 * 3 * 10 * 20 * 96 * 48))
    # 14 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(387072000))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # (8**2* (5 * 3 * 10 * 20 * 96 * 48)) + (8 * (8-1)* (5 * 3 * 10 * 20 * 96 * 48))
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(1658880000))

  def test_conv2d_with_bias(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Conv2D(self.filters, self.kernel, use_bias=True)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # (5 * 3 * 10 * 20 * 96 * 48) * 2 + (20 * 96 * 48)
    self.assertEqual(ef, format_flops(27740160))
    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(27740160))

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
    # (2**2 * (5 * 3 * 10 * 20 * 96 * 48)) + (2 * (2-1)* (5 * 3 * 10 * 20 * 96 * 48)) + (20 * 96 * 48 * 2)
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(83128320))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # (4**2 * (5 * 3 * 10 * 20 * 96 * 48)) + (4 * (4-1)* (5 * 3 * 10 * 20 * 96 * 48)) + (20 * 96 * 48 * 4)
    # 14 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(387440640))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # (8**2 * (5 * 3 * 10 * 20 * 96 * 48)) + (8 * (8-1)* (5 * 3 * 10 * 20 * 96 * 48)) + (20 * 96 * 48 * 8)
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(1659617280)) # (k_h * k_w * c_in * c_out * out_h * out_w) * 2
  
  def test_conv2d_stride(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Conv2D(self.filters, self.kernel, strides=(3, 2), use_bias=True)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # (5 * 3 * 10 * 20 * 96 * 48) * 2 + (20 * 96 * 48)
    self.assertEqual(ef, format_flops(4623360)) 
    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(4623360)) 

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
  #   # (2**2 * (5 * 3 * 10 * 20 * 32 * 24)) + (2 * (2-1)* (5 * 3 * 10 * 20 * 32 * 24)) + (20 * 32 * 24 * 2)
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(13854720))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # N = 2**2
    # N**2 * (k_h * k_w * c_in * c_out * out_h * out_w) 
    # N*(N-1) * (k_h * k_w * c_in * c_out * out_h * out_w) 
    # 14 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(64573440))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # N = 2**3
    # N**2 * (k_h * k_w * c_in * c_out * out_h * out_w) 
    # N*(N-1) * (k_h * k_w * c_in * c_out * out_h * out_w) 
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(276602880))
  
  def test_conv2d_pad_same(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Conv2D(self.filters, self.kernel, padding="SAME", use_bias=False)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # (5 * 3 * 10 * 20 * 100 * 50) * 2 
    self.assertEqual(ef, format_flops(30000000))

    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(30000000))

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
    # (2**2 * (5 * 3 * 10 * 20 * 100 * 50)) + (2 * (2-1)* (5 * 3 * 10 * 20 * 100 * 50))
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(90000000))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # (4**2 * (5 * 3 * 10 * 20 * 100 * 50)) + (4 * (4-1)* (5 * 3 * 10 * 20 * 100 * 50))
    # 14 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(420000000))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # (8**2 * (5 * 3 * 10 * 20 * 100 * 50)) + (8 * (8-1)* (5 * 3 * 10 * 20 * 100 * 50))
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(1800000000))
  
  def test_depthwise_without_bias(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.DepthwiseConv2D(self.kernel, use_bias=False)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # (5 * 3 * 10 * 20 * 96 * 48) * 2
    self.assertEqual(ef, format_flops(1382400))

    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(1382400))

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
    # (2**2 * (5 * 3 * 10 * 96 * 48)) + (2 * (2-1)* (5 * 3 * 10 * 96 * 48))
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(4147200))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # (4**2 * (5 * 3 * 10 * 96 * 48)) + (4 * (4-1)* (5 * 3 * 10 * 96 * 48))
    # 14 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(19353600))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # (8**2 * (5 * 3 * 10 * 96 * 48)) + (8 * (8-1)* (5 * 3 * 10 * 96 * 48))
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(82944000))

  def test_depthwise_with_bias(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.DepthwiseConv2D(self.kernel, use_bias=True)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # ((5 * 3 * 10 * 96 * 48) * 2) + (96 * 48 * 10)
    self.assertEqual(ef, format_flops(1428480))

    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(1428480))

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
    # (2**2 * (5 * 3 * 10 * 96 * 48)) + (2 * (2-1)* (5 * 3 * 10 * 96 * 48)) + ((96 * 48 * 10) * 2)
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(4239360))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # (4**2 * (5 * 3 * 10 * 96 * 48)) + (4 * (4-1)* (5 * 3 * 10 * 96 * 48)) + ((96 * 48 * 10) * 4))
    # 14 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(19537920))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # (8**2 * (5 * 3 * 10 * 96 * 48)) + (8 * (8-1)* (5 * 3 * 10 * 96 * 48)) + ((96 * 48 * 10) * 8))
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(83312640))

  def test_dense_without_bias(self):
    i = tf.keras.layers.Input(100, batch_size=1)
    x = tf.keras.layers.Dense(5, use_bias=False)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # 100 * 5 * 2
    self.assertEqual(ef, format_flops(1000))

    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(1000))

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
    # 2**2 * (100 * 5) + (2 * (2-1) * 100 * 5) 
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(3000))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # 4**2 * (100 * 5) + (4 * (4-1) * 100 * 5)
      # 14 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(14000))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # 8**2 * (100 * 5) + (8 * (8-1) * 100 * 5)
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(60000))

  def test_dense_with_bias(self):
    i = tf.keras.layers.Input(100, batch_size=1)
    x = tf.keras.layers.Dense(5, use_bias=True)(i)

    # Test upstride_type -1
    ef = self.generic_test(i, x, upstride_type=-1)
    # (100 * 5 * 2) + 5
    self.assertEqual(ef, format_flops(1005))

    # Test upstride_type 0
    ef = self.generic_test(i, x, upstride_type=0)
    self.assertEqual(ef, format_flops(1005))

    # Test upstride_type 1
    ef = self.generic_test(i, x, upstride_type=1)
    # 2**2 * (100 * 5) + (2 * (2-1) * 100 * 5) + (5 * 2)
    # 3 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(3010))

    # Test upstride_type 2
    ef = self.generic_test(i, x, upstride_type=2)
    # 4**2 * (100 * 5) + (4 * (4-1) * 100 * 5) + (5 * 4)
      # 14 times more flops upstride type -1 or 0 
    self.assertEqual(ef, format_flops(14020))

    # Test upstride_type 3
    ef = self.generic_test(i, x, upstride_type=3)
    # 8**2 * (100 * 5) + (8 * (8-1) * 100 * 5) + (5 * 8)
    # 60 times more flops upstride type -1 or 0
    self.assertEqual(ef, format_flops(60040))

  def test_keras_layer_relu(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.ReLU()(i)
    
    for upstride_type, num_of_blades in self.map_type.items():
      # For Type -1 and Type 0 
      # eg (100 * 50 * 10) * 2
      # For Type 1, 2 and 3
      # eg ((100 * 50 * 10) * 2) * num_of_blades 
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(100000 * num_of_blades))
    
  def test_activation_relu(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Activation(self.relu)(i)
    
    for upstride_type, num_of_blades in self.map_type.items():
      # For Type -1 and Type 0 
      # eg (100 * 50 * 10) * 2
      # For Type 1, 2 and 3
      # eg ((100 * 50 * 10) * 2) * num_of_blades 
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(100000 * num_of_blades))

  def test_hard_sigmoid(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Activation(self.hard_sigmoid)(i)

    for upstride_type, num_of_blades in self.map_type.items():
      # For Type -1 and Type 0 
      # Relu + one addition + one division
      # eg ((100 * 50 * 10) * 2) + (100 * 50 * 10) *2
      # For Type 1, 2 and 3
      # eg ((100 * 50 * 10) * 2 + (100 * 50 * 10) *2) * num_of_blades
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(200000 * num_of_blades))

  def test_hard_swish(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Activation(self.hard_swish)(i)

    for upstride_type, num_of_blades in self.map_type.items():
      # For Type -1 and Type 0 
      # hard_sigmoid * x
      # For Type 1, 2 and 3
      # eg (hard_sigmoid * x) * num_of_blades
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(250000 * num_of_blades))

  def test_max_pooling(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(i)

    for upstride_type, num_of_blades in self.map_type.items():
      # For Type -1 and Type 0 
      # pool_size * output_shape
      # eg (49 * 24 * 10 * 3 * 3))
      # For Type 1, 2 and 3
      # eg (49 * 24 * 10 * 3 * 3) * num_of_blades
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(105840 * num_of_blades))

  def test_average_pooling(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2))(i)

    for upstride_type, num_of_blades in self.map_type.items():
      # For Type -1 and Type 0 
      # pool_size * output_shape
      # eg (49 * 24 * 10 * 3 * 3))
      # For Type 1, 2 and 3
      # eg (49 * 24 * 10 * 3 * 3) * num_of_blades
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(105840 * num_of_blades))

  def test_global_max_pooling(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.GlobalMaxPooling2D()(i)

    for upstride_type, num_of_blades in self.map_type.items():
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(50000 * num_of_blades)) # (100 * 50 * 10) * num_of_blades

  def test_global_avg_pooling(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.GlobalAveragePooling2D()(i)

    for upstride_type, num_of_blades in self.map_type.items():
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(50000 * num_of_blades)) # (100 * 50 * 10) * num_of_blades
  
  def test_add(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Add()([i, i])

    for upstride_type, num_of_blades in self.map_type.items():
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(50000 * num_of_blades)) # (100 * 50 * 10) * num_of_blades

  def test_multiply(self):
    i = tf.keras.layers.Input(self.input_shape, batch_size=1)
    x = tf.keras.layers.Multiply()([i, i])
    for upstride_type, num_of_blades in self.map_type.items():
      # (num_of_blades is 1, 1, 2, 4, 8 for upstride_type -1, 0, 1, 2 and 3 respectively)
      ef = self.generic_test(i, x, upstride_type=upstride_type)
      self.assertEqual(ef, format_flops(50000 * num_of_blades)) # (100 * 50 * 10) * num_of_blades
