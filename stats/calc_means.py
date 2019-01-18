from sys import argv, stderr
import pandas as pd
import numpy as np


def main(args):

    n = 2000
    m = 100

    matrix = np.zeros((n, m))
    means = np.zeros(n)

    for col in range(m):
        path = '~/workspace/intra/intra-distances_128_%i.csv' % col
        df = pd.read_csv(path, header=None)
        matrix[:, col] = np.array(df.values[:n])[:, 0]
    for line in range(n):
        means[line] = np.mean(matrix[line, :])

    name = 'intra_128.csv'
    f = open(name, 'w')
    for value in means:
        f.write(str(value) + '\n')
    f.close()


if __name__ == '__main__':
    main(argv)
