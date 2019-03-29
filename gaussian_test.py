"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-21 下午10:20 
  description:
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d

x, y = np.mgrid[-2:2:200j, -2:2:200j]
z=(1/2*math.pi*3**2)*np.exp(-(x**2+y**2)/2*3**2)
print(z.shape)
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)#绘面

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()