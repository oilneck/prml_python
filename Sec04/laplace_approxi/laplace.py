import numpy as np
from base_module import *
import matplotlib.pyplot as plt
from scipy import integrate
#%matplotlib inline

def pdf(z):
    gaussian_pdf = Gaussian_Feature(variance=1).gauss_function(z,mean=0)
    sigmoid_pdf = Sigmoid_Feature(std=1).sigmoid_function(20 * z + 4,mean=0)
    return gaussian_pdf * sigmoid_pdf



z = np.linspace(-2,20,470)
nomalize_facter = integrate.quad(pdf, -np.inf, np.inf)[0]
pz = pdf(z) / nomalize_facter

log_fz = np.log(pdf(z))
z0_index = np.argmax(pdf(z))
z0 = z[z0_index]# mode
A = -np.gradient(np.gradient(log_fz,z),z)[z0_index]
qz = np.sqrt(0.5 * A /np.pi) * np.exp(-0.5 * A * (z-z0)**2)

plt.figure(figsize=(7, 3))
plt.subplot(1,2,1)
plt.fill_between(z,pz,color='#e5cf5c')
plt.plot(z,qz,color='r',linewidth=1)
plt.xlim(-2,4)
plt.ylim(0,0.8)
plt.yticks(np.arange(0,1,0.2))

plt.subplot(1,2,2)
plt.plot(z,-np.log(pz),color='#e5cf5c',linewidth=1)
plt.plot(z,-np.log(qz),color='r',linewidth=1)
plt.xlim(-2,4)
plt.ylim(0,40)
plt.yticks(np.arange(0,41,10))
plt.tick_params(left=True, right=True)


plt.tight_layout()
plt.show()
