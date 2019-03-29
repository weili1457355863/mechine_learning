"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-20 下午4:31 
  description: show gaussian distibution
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from mpl_toolkits.mplot3d import Axes3D
# from scipy import stats
# mu_params = [-1, 0, 1]
# sd_params = [0.5, 1, 1.5]
# x = np.linspace(-7, 7, 100)
# f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, figsize=(12,8))
# for i in range(3):
#     for j in range(3):
#         mu = mu_params[i]
#         sd = sd_params[j]
#         y = stats.norm(mu, sd).pdf(x)
#         ax[i, j].plot(x, y)
#         ax[i, j].plot(0,0, label='mu={:3.2f}\nsigma={:3.2f}'.format(mu,sd), alpha=0)
#         ax[i, j].legend(fontsize=10)
# ax[2,1].set_xlabel('x', fontsize=16)
# ax[1,0].set_ylabel('pdf(x)', fontsize=16)
# plt.suptitle('Gaussian PDF', fontsize=16)
# plt.tight_layout()
# plt.show()
len = 8
step = 0.4
def build_gaussian_layer(mean, standard_deviation):
    x = np.arange(-len, len, step)
    x=np.reshape(x,(x.shape[0],1))
    y = np.arange(-len, len, step)
    y=np.reshape(y,(y.shape[0],1))
    z=np.zeros((x.shape))
    # x, y = np.meshgrid(x, y)
    # print(x)
    # z = np.exp(-((y-mean)**2 + (x - mean)**2)/(2*(standard_deviation**2)))
    # z = z/(np.sqrt(2*np.pi)*standard_deviation)
    # print(z.shape)
    X=np.concatenate((x,y),axis=1)
    # print(mean)
    i=0
    for x in zip(X):
        print(x)
        x=np.reshape(x,(2,1))
        mean=np.reshape(mean,(2,1))
        z[i] =exp(-0.5*(np.dot(np.dot((x-mean).T,np.linalg.inv(standard_deviation)),(x-mean))))
        z[i]=z[i]/(2*np.pi*np.sqrt(np.linalg.det(standard_deviation)))
        i=i+1
    x, y = np.meshgrid(x.T, y.T)
    # print("z.shape:",z.shape)
    return (x, y, z)

fig = plt.figure()
ax = fig.gca(projection='3d')

mean=np.array([0,1])
standard_deviation=np.array([[1,0],[0,1]])
print(standard_deviation.shape)
print(mean.shape)
x3, y3, z3 = build_gaussian_layer(mean, standard_deviation)
print(z3)
ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='rainbow')
plt.show()