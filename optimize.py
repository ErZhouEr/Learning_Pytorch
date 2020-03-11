
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # 3d is unable to show without this tool

def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x=np.arange(-6,6,0.1)
y=np.arange(-6,6,0.1)
X,Y=np.meshgrid(x,y)
print(X.shape,Y.shape)

z=himmelblau([X,Y])
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(X,Y,z)   # z is a 2-dimension vector now
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x=torch.tensor([0.,0.],requires_grad=True)
optimizor=torch.optim.Adam([x],lr=1e-3)
for step in range(20000):
    pred=himmelblau(x)
    optimizor.zero_grad()
    pred.backward()
    optimizor.step()
    if step%2000==0:
        print(step,x.tolist(),pred.item())

