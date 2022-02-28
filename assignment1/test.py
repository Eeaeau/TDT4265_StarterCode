import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# x = np.linspace(5, 8, 4)
# z = np.arange(4)
# y = np.transpose(x)
# # print("x", x)
# # print("z", z)
# # # print(y)

# batch_size = 20
# classes = 1

# m = np.random.rand(batch_size, classes)
# n = np.random.rand(batch_size, classes)

# print("m", m.shape)
# print("n", n.shape)

# # print(m.T @ n)
# print(np.tensordot(m.T, n))

d = deque(maxlen=10)

d.append(2)
d.append(1)
d.append(3)

print(d > 1)

# x2 = x.reshape((4, 1))
# z2 = z.reshape((4, 1))

# print("x2: ", x2)
# print("z2: ", z2)
# print(np.multiply(x2, z2).sum(axis=0))
# print(np.matmul(x2.T, z2))
# print((x2 + z2))
# print(np.linalg.multi_dot([x2, z2.T]))

# rng = np.random.default_rng()
# arr = np.arange(10)
# print(arr)
# rng.shuffle(arr, axis=0)

# print(arr)
