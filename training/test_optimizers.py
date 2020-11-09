import unittest
from . import optimizers

import numpy as np


class TestLRDecay(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.epochs = np.arange(0, 100)
    cls.inital_learning_rate = 0.045

    cls.lr_params = {
        'patience': 4,
        'factor': 0.2,
        'min_lr': 0.00001,
        'early_stopping': 10,
        'drop_rate': 0.5,
        'drop_after_num_epoch': 10,
        'drop_schedule': [10, 40, 20, 40, 50, 90],
        'power': 5.0,
        'decay_rate': 0.5,
        'decay_step': 1.0,
        'alpha': 0.01,
        'staircase': False,
        'lr_list': [1., 2., 3., 4., 5., 6.]
    }

  def non_increasing(self, decay):
    """Checks if all the learning rates are non increasing over the total epochs

    Args:
        decay (instance): Instance of the decay to be tested

    Returns:
        bool : Compares ith and i + 1st element of the value_list
        returns True if all i >= i+1 else False
    """
    value_list = [decay(i) for i in self.epochs]
    return all([i >= j for i, j in zip(value_list, value_list[1:])])

  def test_exponential_decay(self):
    decay = optimizers.ExponentialDecay(self.inital_learning_rate)

    # check initial value and higher epoch to ensure the values are matching
    self.assertEqual(decay(0), 0.045)  # keras model.fit starts from epoch 0
    self.assertEqual(decay(1), 0.0405)
    self.assertAlmostEqual(decay(3).numpy(), 0.03280, places=5)
    self.assertAlmostEqual(decay(10).numpy(), 0.01569, places=5)

    # test function is not increasing over number of epochs
    self.assertTrue(self.non_increasing(decay), True)

    # check value doesn't decrease to zero at the end of epoch
    self.assertNotEqual(decay(self.epochs[-1]), 0.0)

  def test_step_decay(self):
    decay = optimizers.StepDecay(self.inital_learning_rate,
                      self.lr_params["drop_rate"],
                      self.lr_params["drop_after_num_epoch"])

    # check initial value and higher epoch to ensure the values are matching
    self.assertEqual(decay(0), 0.045)
    self.assertEqual(decay(9), 0.045)
    self.assertEqual(decay(19), 0.0225)

    # test function is not increasing over number of epochs
    self.assertTrue(self.non_increasing(decay), True)

    # check value doesn't decrease to zero at the end of epoch
    self.assertNotEqual(decay(self.epochs[-1]), 0.0)

  def test_step_decay_schedule(self):
    decay = optimizers.StepDecaySchedule(self.inital_learning_rate,
                              self.lr_params['drop_schedule'],
                              self.lr_params["drop_rate"],
                              self.epochs[-1])

    # check initial value and higher epoch to ensure the values are matching
    self.assertEqual(decay(0), 0.045)
    self.assertEqual(decay(9), 0.045)
    self.assertEqual(decay(11), 0.045 * 0.5)
    self.assertEqual(decay(21), 0.045 * 0.5 * 0.5)
    self.assertAlmostEqual(decay(51), 0.045 * (0.5**4), places=4)

    # test function is not increasing over number of epochs
    self.assertTrue(self.non_increasing(decay), True)

    # check value doesn't decrease to zero at the end of epoch
    self.assertNotEqual(decay(self.epochs[-1]), 0.0)

  def test_polynomial_decay(self):
    decay = optimizers.PolynomialDecay(self.inital_learning_rate,
                            self.lr_params["power"],
                            self.epochs[-1])

    # check initial value and higher epoch to ensure the values are matching
    self.assertEqual(decay(0), 0.045)
    self.assertAlmostEqual(decay(5).numpy(), 0.03482, places=3)

    # test function is not increasing over number of epochs
    self.assertTrue(self.non_increasing(decay), True)

    # check value doesn't decrease to zero at the end of epoch
    self.assertNotEqual(decay(self.epochs[-1]), 0.0)

  def test_inverse_time_decay(self):
    for flag in [True, False]:
      decay = optimizers.InverseTimeDecay(self.inital_learning_rate,
                               self.lr_params["decay_rate"],
                               self.lr_params["decay_step"],
                               flag)

      # check initial value and higher epoch to ensure the values are matching
      self.assertEqual(decay(0), 0.045)
      if flag:
        self.assertAlmostEqual(decay(9).numpy(), 0.00900, places=5)
        self.assertAlmostEqual(decay(29).numpy(), 0.003, places=3)
      else:
        self.assertAlmostEqual(decay(9), 0.00818, places=5)
        self.assertAlmostEqual(decay(29), 0.00290, places=5)

      # test function is not increasing over number of epochs
      self.assertTrue(self.non_increasing(decay), True)

      # check value doesn't decrease to zero at the end of epoch
      self.assertNotEqual(decay(self.epochs[-1]), 0.0)

  def test_cosine_decay(self):
    decay = optimizers.CosineDecay(self.inital_learning_rate,
                        self.lr_params["alpha"],
                        self.epochs[-1])

    # check initial value and higher epoch to ensure the values are matching
    self.assertEqual(decay(0), 0.045)
    self.assertAlmostEqual(decay(10).numpy(), 0.04390, places=3)
    self.assertAlmostEqual(decay(60).numpy(), 0.01584, places=3)

    # test function is not increasing over number of epochs
    self.assertTrue(self.non_increasing(decay), True)

    # check value doesn't decrease to zero at the end of epoch
    self.assertNotEqual(decay(self.epochs[-1]), 0.0)

  def test_explicit_decay(self):
    decay = optimizers.ExplicitSchedule(self.inital_learning_rate,
                                        self.lr_params['drop_schedule'],
                                        self.lr_params['lr_list'],
                                        )
    self.assertEqual(decay(0), 0.045)
    for i in range(1, 10):
      self.assertEqual(decay(i), 0.045)
    for i in range(10, 40):
      self.assertEqual(decay(i), 1.)

    # test empty list
    decay = optimizers.ExplicitSchedule(self.inital_learning_rate, [], [], )
    with self.assertRaises(AssertionError):                                        
      decay(0)

    # test list with a single element
    decay = optimizers.ExplicitSchedule(self.inital_learning_rate,
                                        [0],
                                        [0.1],
                                        )
    

    self.assertEqual(decay(0), 0.1)
    for i in range(1, 10):
      self.assertEqual(decay(i), 0.1)  
    
