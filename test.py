import numpy as np
import matplotlib.pyplot as plt

x = np.zeros([5, 5])
x[2, 2] = 1

label_x = range(-2, 3)
label_y = range(-2, 3)

# temp = ['{:d}'.format(x) for x in label_x]
# temp = ['-2', '-1', '0']
temp = list(label_x)
temp.insert(0, temp[0])

fig, ax = plt.subplots()
plt.imshow(x)
ax.set_xticklabels(temp)
ax.set_yticklabels(temp)
plt.show()
