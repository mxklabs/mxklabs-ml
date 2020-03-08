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
    
  def train_discriminator(self, epochs=100, num_samples=100, verbose=1):
    fake_x = self._generator.predict(self._input_sampler(num_samples))
    fake_y = np.full(shape=(num_samples, 1), fill_value=0)

    fake_x_val = self._generator.predict(self._input_sampler(max(num_samples//4, 1)))
    fake_y_val =  np.full(shape=(max(num_samples//4, 1), 1), fill_value=0)

    x = np.concatenate((self._data.real_x, fake_x))
    y = np.concatenate((self._data.real_y, fake_y))

    x_val = np.concatenate((self._data.real_x_val, fake_x_val))
    y_val = np.concatenate((self._data.real_y_val, fake_y_val))

    self._discriminator.fit(x, y, validation_data=(x_val, y_val), epochs=epochs, verbose=verbose)

  def train_generator(self, epochs=100, num_samples=100, verbose=1):  
    fake_x = self._input_sampler(num_samples)
    fake_y = np.full(shape=(num_samples, 1), fill_value=1)
    
    fake_x_val = self._input_sampler(max(num_samples//4, 1))
    fake_y_val = np.full(shape=(max(num_samples//4, 1), 1), fill_value=1)
    
    self._gan_model.fit(fake_x, fake_y, validation_data=(fake_x_val, fake_y_val), epochs=epochs, verbose=verbose)
  
  def train(self, iterations=100, epochs_per_round=1, num_samples=100, 
            train_generator_only=False,
            train_discriminator_only=False, verbose=1, callback=None):

    for i in range(iterations):
      input_ = self._input_sampler(num_samples)
      gen_output = self._generator.predict(input_)
      dis_output = self._discriminator.predict(gen_output)
      mean_disc_score = dis_output.mean()

      if callback is not None:
        callback(gan_model=self._gan_model, 
                 generator=self._generator, 
                 discriminator=self._discriminator, 
                 generator_output=gen_output, 
                 discriminator_output=dis_output, 
                 score=mean_disc_score)
      
      print(f"---------- Round {i} predictions: {mean_disc_score} ----------" )
      #if i % plot_period == 0:
      #plot_data(discriminator, gen_output)

      if train_discriminator_only or (not train_generator_only and mean_disc_score >= 0.5):
        print(f"Mean score is {mean_disc_score} -- training discriminator.")
        self.train_discriminator(epochs=epochs_per_round, num_samples=num_samples, verbose=verbose)
      else:
        print(f"Mean score is {mean_disc_score} -- training generator.")
        self.train_generator(epochs=epochs_per_round, num_samples=num_samples, verbose=verbose)  

  
"""
if __name__ == '__main__':
  
  NUM_SAMPLES=100
  def gen_data(num_samples):
    noise = np.array([random.normalvariate(0, 0.4) for _ in range(num_samples)])
    x1 = np.array([random.random() * 2 - 1 for _ in range(num_samples)])
    x2 = np.array((np.power(x1*2, 3) + np.sin(x1*12 - 2)*0.7 + noise)/7)
    y = np.full(shape=(num_samples, 1), fill_value=1)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    return np.concatenate((x1, x2), axis=-1), y
    
  real_x, real_y = gen_data(NUM_SAMPLES)
  real_x_val, real_y_val = gen_data(NUM_SAMPLES//4)

  
  latent_shape = (5,)
  data_shape = (2,) # data shape
  num_samples = 40

  # Get real words for training.
  latent_space = tf.keras.layers.Input(shape=latent_shape)

  # Using Leaky RELU because of tip https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/.

  # Discriminator input is (n, 21, 27) output is (n, 1)
  discriminator = tf.keras.models.Sequential(name="discriminator")
  discriminator.add(tf.keras.layers.Dense(name='dhidden1', units=50, use_bias=True,
                                          activation=tf.keras.layers.LeakyReLU(alpha=0.3), input_shape=data_shape))
  discriminator.add(tf.keras.layers.Dense(name='dhidden2', units=50, use_bias=True,
                                          activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
  discriminator.add(tf.keras.layers.Dense(name='dhidden3', units=15, use_bias=True,
                                          activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
  discriminator.add(tf.keras.layers.Dense(name='doutput', units=1, use_bias=True, activation='tanh'))

  #discriminator.add(tf.keras.layers.Dense(name='dhidden1', units=50, use_bias=True, activation='tanh', input_shape=data_shape))
  #discriminator.add(tf.keras.layers.Dense(name='dhidden2', units=50, use_bias=True, activation='tanh'))
  #discriminator.add(tf.keras.layers.Dense(name='dhidden3', units=15, use_bias=True, activation='tanh'))
  #discriminator.add(tf.keras.layers.Dense(name='doutput', units=1, use_bias=True, activation='linear'))
  discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])

  # Generator.
  discriminator.trainable = False

  generator = tf.keras.models.Sequential(name="generator")
  generator.add(tf.keras.layers.Dense(name='ghidden1', units=15, use_bias=True, activation='tanh', input_shape=latent_shape))
  generator.add(tf.keras.layers.Dense(name='ghidden2', units=15, use_bias=True, activation='tanh'))
  generator.add(tf.keras.layers.Dense(name='gout', units=2, use_bias=True, activation='linear'))

  gan_model = tf.keras.models.Model(latent_space, discriminator(generator(latent_space)))
  gan_model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='mse', metrics=['binary_accuracy'])

  gan = Gan(gan_model, real_x, real_y, real_x_val, real_y_val)
  gan.train(iterations=5, epochs_per_round=1, num_samples=1, train_discriminator_only=True, verbose=1)
"""