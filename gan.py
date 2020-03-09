import collections
import numpy as np
import random
import tensorflow as tf

Data = collections.namedtuple('Data', ['real_x', 'real_y', 'real_x_val', 'real_y_val'])

class Gan:
        
  def __init__(self, gan_model, data,
               input_sampler=None):
    self._gan_model = gan_model
    self._generator = gan_model.get_layer("generator")
    self._discriminator = gan_model.get_layer("discriminator")
    self._data = data
    
    assert(self._data.real_x.shape[1:] == self._discriminator.input_shape[1:])
    assert(self._data.real_x_val.shape[1:] == self._discriminator.input_shape[1:])
    
    if input_sampler is not None:
      self._input_sampler = input_sampler
    else:
      self._input_sampler = lambda num_samples: np.random.normal(0, 1, (num_samples, *self._generator.input_shape[1:]))
  
  def _make_discriminator_data(self, fake_x, num_samples=100):
    fake_y = np.full(shape=(num_samples, 1), fill_value=0)
    x = np.concatenate((self._data.real_x, fake_x))    
    y = np.concatenate((self._data.real_y, fake_y))    
    return x, y
    
  def _make_generator_data(self, input_, num_samples=100):
    y = np.full(shape=(num_samples, 1), fill_value=1)
    return input_, y
  
  def train(self, iterations=100, epochs_per_round=1, num_samples=100, 
            train_generator_only=False,
            train_discriminator_only=False, verbose=1, callback=None):

    for i in range(iterations):
      input_ = self._input_sampler(num_samples)
      gen_output = self._generator.predict(input_)
      dis_output = self._discriminator.predict(gen_output)
            
      dx, dy = self._make_discriminator_data(gen_output, num_samples)
      
      scores = self._discriminator.evaluate(x=dx, y=dy)[1]
      score = scores.mean()
      #mean_disc_score = dis_output.mean()

      if callback is not None:
        callback(gan_model=self._gan_model, 
                 generator=self._generator, 
                 discriminator=self._discriminator, 
                 generator_output=gen_output, 
                 discriminator_output=dis_output, 
                 score=score)
      
      print(f"---------- Round {i} predictions: {score} ----------" )
      #if i % plot_period == 0:
      #plot_data(discriminator, gen_output)

      if train_discriminator_only or (not train_generator_only and score < 0.70):
        print(f"Mean score is {score} -- training discriminator.")
        self._discriminator.fit(dx, dy, epochs=epochs_per_round, verbose=verbose)
      else:
        print(f"Mean score is {score} -- training generator.")
        gx, gy = self._make_generator_data(input_, num_samples)      
        self._gan_model.fit(gx, gy, epochs=epochs_per_round, verbose=verbose)
