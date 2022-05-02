import matplotlib.pyplot as plt
import numpy as np

path_to_prefix = "evaluation/moses_sparse_ppe/eval_"

epochs = np.arange(1,11)
acc = []
rec = []
f = []

for i in range(1, 11):
    path = path_to_prefix + str(i)
    with open(path, "r") as file:
        lines = file.readlines()
        scores = [float(j) for j in lines[3].split()]
        acc.append(scores[3])
        rec.append(scores[4])
        f.append(scores[5])
        print(scores)

plt.plot(epochs, acc, 'ko')
plt.ylabel('Accuracy')

plt.plot(epochs, rec, 'co')
plt.ylabel('Recall')

plt.plot(epochs, f, 'mo')
plt.ylabel('F0.5')
