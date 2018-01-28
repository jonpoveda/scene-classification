import matplotlib

# Force matplotlib to not use any Xwindows backend. If you need to import
# pyplot, do it after setting `Agg` as the backend.
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

cm = np.array([[76, 3, 4, 0, 20, 11, 0, 2],
               [1, 68, 1, 8, 6, 9, 3, 5],
               [5, 1, 56, 3, 4, 4, 3, 0],
               [3, 6, 2, 50, 7, 9, 9, 8],
               [10, 2, 5, 12, 63, 12, 3, 7],
               [8, 11, 4, 4, 9, 79, 2, 1],
               [0, 2, 1, 8, 7, 1, 58, 3],
               [2, 3, 1, 9, 9, 2, 1, 81]])

cm = np.array([[37, 3, 3, 0, 0, 4, 0, 0],
               [2, 42, 0, 1, 0, 1, 0, 0],
               [2, 0, 36, 0, 0, 2, 0, 0],
               [2, 0, 1, 23, 1, 0, 2, 1],
               [1, 0, 1, 0, 26, 0, 5, 4],
               [3, 3, 1, 0, 0, 37, 0, 1],
               [0, 0, 0, 1, 3, 0, 26, 2],
               [0, 0, 0, 0, 2, 5, 1, 35]])

# print(np.sum(cm, axis=0))
print(np.sum(cm, axis=1))
norm_cm = np.divide(cm, np.sum(cm, axis=1), dtype=np.float32)
print(np.round(norm_cm, decimals=2))
plt.matshow(norm_cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label (100%)')
plt.xlabel('Predicted label')
plt.show()
plt.savefig('cm.jpg')
