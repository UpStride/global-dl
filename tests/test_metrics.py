import unittest
import numpy as np
import tensorflow as tf
from training.metrics import count_flops,  count_trainable_params, InformationDensity, NetScore, count_flops_efficient, get_type
from upstride.type2.tf.keras import layers as type2_layers
from upstride.type1.tf.keras import layers as type1_layers

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

def relu(x):
  return tf.nn.relu(x)

def hard_sigmoid(x):
  return tf.nn.relu6(x + 3.) / 6.

def hard_swish(x):
  return x * hard_sigmoid(x)
class TestCountFlops(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.input = tf.keras.layers.Input((32, 32, 3), batch_size=1)

  def generic_test(self, output, upstride_type, get_count_flops=False):
    layers, _ = get_type(upstride_type)
    if upstride_type > 0: 
      up_in = layers.TF2Upstride()(self.input)
      out = output(up_in)
    else:
      out = output(self.input)
    model = tf.keras.Model(self.input, out) 
    efficient_count = count_flops_efficient(model, upstride_type)
    if get_count_flops:
      count = count_flops(model)
      return count, efficient_count
    else:
      return efficient_count
      
  def test_conv2d_no_bias(self):
    x = tf.keras.layers.Conv2D(16, (3, 3), use_bias=False)
    ef = self.generic_test(x, upstride_type=-1)
    self.assertTrue(ef, 486000) # (k_h * k_w * c_in * c_out * out_h * out_w) * 2

  def test_conv2d_no_bias_type2(self):
    x = type2_layers.Conv2D(16 // 4, (3, 3), use_bias=False)
    ef = self.generic_test(x, upstride_type=2)
    print(ef)
    # N**2 * (k_h * k_w * c_in * c_out * out_h * out_w) + N*(N-1) (k_h * k_w * c_in * c_out * out_h * out_w)
    self.assertEqual(ef, 2721600)  

  def test_conv2d_bias(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3))(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_conv2d_strides(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2))(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_conv2d_padding_same(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_conv2d_relu(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_relu(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.ReLU()(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model) # return 0...
    ef = count_flops_efficient(model, upstride_type=-1)
    print(ef)
    self.assertEqual(ef, 301056)

  def test_hard_sigmoid(self):
    i = tf.keras.layers.Input((1), batch_size=1)
    x = tf.keras.layers.Activation(hard_sigmoid)(i)
    model = tf.keras.Model(i, x)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertEqual(ef, 4) # 2 for Relu (min, max) 1 addition and 1 multiplication

  def test_hard_swish(self):
    i = tf.keras.layers.Input((1), batch_size=1)
    x = tf.keras.layers.Activation(hard_swish)(i)
    model = tf.keras.Model(i, x)
    ef = count_flops_efficient(model, upstride_type=-1)
    print(ef)
    self.assertEqual(ef, 8) # 2 * hard_sigmoid 

  def test_max_pooling(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertEqual(f, ef)

  def test_global_max_pooling(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.GlobalMaxPooling2D()(i)
    model = tf.keras.Model(i, x)
    # f = count_flops(model) # returns 0
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertEqual(ef, 150528) # 224 * 224 * 3

  def test_global_avg_pooling(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.GlobalAveragePooling2D()(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model) 
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertEqual(f, ef) 
  
  def test_add(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Add()([i, i])
    model = tf.keras.Model(i, x)
    # f = count_flops(model) # returns 0
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertEqual(ef, 150528) # 224 * 224 * 3

  def test_multiply(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Multiply()([i, i])
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertEqual(f, ef)

  def test_dense_no_biases(self):
    i = tf.keras.layers.Input((1000), batch_size=1)
    x = tf.keras.layers.Dense(100, use_bias=False)(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model, upstride_type=-1)
    # f = 400001 ??? I have no explanation for this
    # ef = 200000
    self.assertEqual(ef, 200000) # 1000 * 100 * 2 

  def test_dense_biases(self):
    i = tf.keras.layers.Input((1000), batch_size=1)
    x = tf.keras.layers.Dense(100)(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertEqual(ef, 200100) # (1000 * 100 * 2) + 100

  def test_depthwiseconv2d(self):
    i = tf.keras.layers.Input((32, 32, 3), batch_size=1)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), use_bias=False)(i)
    model = tf.keras.Model(i, x)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertTrue(ef, (48600)) # 30 * 30 * 3 * 3 * 3 * 2

  def test_depthwiseconv2d_bias(self):
    i = tf.keras.layers.Input((32, 32, 3), batch_size=1)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), use_bias=True)(i)
    model = tf.keras.Model(i, x)
    ef = count_flops_efficient(model, upstride_type=-1)
    self.assertTrue(ef, (48600)) # (30 * 30 * 3 * 3 * 3 * 2) + (3 * 3 * 3)
