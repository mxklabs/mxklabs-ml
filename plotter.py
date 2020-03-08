import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc

class GanPlotter:
  
  def __init__(self):
    self._fig = plt.figure()
    N = 4
    ax = plt.axes(xlim=(0, 2), ylim=(0, 100))
    
    self._lines = [plt.plot([], [])[0] for _ in range(N)] #lines to animate
    self._rectangles = plt.bar([0.5,1,1.5],[50,40,90],width=0.1) #rectangles to animate
    self._patches = self._lines + list(self._rectangles) #things to animate
  
  def init(self):

    #init lines
    for line in self._lines:
      line.set_data([], [])

    #init rectangles
    for rectangle in self._rectangles:
      rectangle.set_height(0)

    return self._patches #return everything that must be updated

  def animate(self, i):
    #animate lines
    for j,line in enumerate(self._lines):
      line.set_data([0, 2], [10 * j,i])#
      
    #animate rectangles
    for j,rectangle in enumerate(self._rectangles):
      rectangle.set_height(i/(j+1))
  
    return self._patches #return everything that must be updated
  
  def add_frame(self):

  def as_html(self):
    anim = animation.FuncAnimation(self._fig, self.animate, init_func=self.init,
                                   frames=100, interval=100, blit=True)

    # Note: below is the part which makes it work on Colab
    rc('animation', html='jshtml')
    return anim
  