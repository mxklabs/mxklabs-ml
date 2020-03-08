import functools
import numpy as np
import random
import tensorflow as tf

from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection

from gan import Gan
import pltanimation as plta

def plot_init(fig):
  ax = fig.add_subplot(1,1,1)
  ax.set_title("Empty plot")
  #plot1 = ax.plot([], [])[0]
  generator_out = ax.scatter([], [])
  
  return [generator_out]

def animation_callback(ani, gan_model, generator, discriminator, generator_output, discriminator_output, score):
  for i in range(10):
    xy1 = np.random.uniform(-1, 1, (20, 2))
    xy2 = np.random.uniform(-1, 1, (100, 2))
    #scatter1_data = ((xy[:,0], xy[:,1]), {})
    ani.add_frame([
        #(Line2D.set_data, xy1[:,0], xy1[:,1]),
        (PathCollection.set_offsets, generator_output)
    ])
  
if __name__ == '__main__':
  
  ani = plta.PltAnimation(plot_init, figsize=(5, 5))     
  
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
  gan.train(iterations=5, epochs_per_round=1, num_samples=1, train_discriminator_only=True, verbose=1, callback=functools.partial(callback, animation_callback=ani))
  
  ani.as_html()
  