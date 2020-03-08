import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc

class PltAnimation:
  
  def __init__(self, plot_init, **figure_kwargs):
    self._plot_init = plot_init    
    self._figure_kwargs = figure_kwargs
    self._drawables = []
    self._frames = []
    self._fig = plt.figure(**self._figure_kwargs)
  
  def add_frame(self, data):
    self._frames.append(data)
    
  def _init(self):
    self._drawables = self._plot_init(self._fig)
    return self._drawables

  def _animate(self, i):
    for d in range(len(self._drawables)):
        self._frames[i][d][0](self._drawables[d], *self._frames[i][d][1:])
    return self._drawables

  def as_html(self, interval=100):
    anim = animation.FuncAnimation(self._fig, self._animate, 
                                   init_func=self._init,
                                   frames=len(self._frames), interval=interval, blit=True)
    rc('animation', html='jshtml')
    plt.close(self._fig)
    return anim
  