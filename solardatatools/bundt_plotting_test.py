import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_bundt_cake
np.random.seed(42)
test_data = np.random.rand(365, 100)
fig = plot_bundt_cake(test_data)
if fig:
    plt.show()
