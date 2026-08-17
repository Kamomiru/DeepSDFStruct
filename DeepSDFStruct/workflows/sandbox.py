from DeepSDFStruct.sdf_primitives import CrossMsSDF
from DeepSDFStruct.deep_sdf.plotting import *
import matplotlib.pyplot as plt

SDF = CrossMsSDF(0.5)

radii = [0.1 * radius for radius in range(1, 10, 1)]
print(radii)

fig, axes = plt.subplots(
    3,3,
    figsize=(12, 12),
    squeeze= False)

axes = axes.flatten()

for radius, ax in zip(radii, axes):
    SDF.setRadius(radius)
    SDF.plot_slice(normal=(0,1,0), ax = ax)

    # Add title to each subplot
    ax.set_title(
        f"Radius = {radius:.1f}",
        fontsize=14,
        pad=10
    )

fig.savefig("C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/workflows/testPlot.png")








