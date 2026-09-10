from DeepSDFStruct.deep_sdf.GAN_helpers.gan_training_helpers import *
from DeepSDFStruct.deep_sdf.plotting import *
from matplotlib import pyplot as plt
from DeepSDFStruct.deep_sdf.GAN_helpers.chi3d_SDF import Chi3DPrismSDF
import torch
from DeepSDFStruct.deep_sdf.plotting import *

rows = 4
colums = 4
fig, axes = plt.subplots(rows,colums, figsize = (4,4))

"""
phi: [0.0, -pi/4]
t: [0.05, 0,15]
x1: [0.25, 1.57]
x2: [0.01] ([0.01, 0.2])
r: [0.01]
"""
t = 0.15
x2 = 0.01
r = 0.1
for i in range(0,rows):
    x1 = 0.25 + 0.44 * i
    
    for j in range(0,colums):
        phi = 0.0 - torch.pi/4/(colums -1) * (j)
        # [phi, t, x1, x2, r]
        params = torch.tensor([phi, t, x1, x2, r])
        param_tensor = params.expand(5, -1)
        #print(param_tensor)

        sdf = Chi3DPrismSDF(param_tensor)

        ax = sdf.plot_slice(origin=(0, 0, 0), normal=(0, 1, 0), ax = axes[i, j])
        axes[i, j].set_title(f"x1: {x1:.2f} | phi: {phi:.2f}")
        

plt.show()



