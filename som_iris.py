import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Make some illustrative fake data:

x = np.arange(0, np.pi, 0.1)
y = np.arange(0, 2 * np.pi, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) * np.sin(Y) * 10

colors = [(0.7, 0, 0), (0, 0, 0)]  # R -> G -> B
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'my_list'

cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
# Fewer bins will result in "coarser" colomap interpolation
plt.imshow(Z, origin='lower', cmap=cmap)
plt.colorbar()
plt.show()
