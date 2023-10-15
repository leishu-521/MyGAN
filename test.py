import numpy as np
import matplotlib.pyplot as plt

# point = np.random.randn(3) * .02
# print(point)

N_POINTS = 128
RANGE = 3
plt.clf()

a = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
print(a.shape)


# points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
# points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
# points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
# points = points.reshape((-1, 2))