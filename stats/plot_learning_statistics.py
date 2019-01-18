import pandas as pd
from stats.compute_stats import *

# Example
accuracies = np.array([
    [0.95, 0.90, 0.89, 0.60],
    [0.99, 0.98, 0.90, 0.65],
    [0.95, 0.92, 0.89, 0.75],
    [0.98, 0.96, 0.85, 0.69],
    [0.97, 0.93, 0.88, 0.63],
    [0.96, 0.93, 0.93, 0.79],
    [0.99, 0.99, 0.89, 0.78],
    [0.99, 0.99, 0.98, 0.80],
    [0.98, 0.97, 0.89, 0.66],
    [0.99, 0.96, 0.95, 0.51],
])

raw_distances = np.array(
    [.51, .55, .53, .6, .68, .55, .78, .53, .5, .53]
)


# Executions
#plot_individual_learning_rates(accuracies)
"""
# plot stability vs noisiness
path1 = '~/workspace/some_data/intra/results/intra_128.csv'
path2 = '~/workspace/some_data/intra/results/intra_64.csv'
path3 = '~/workspace/some_data/intra/results/intra_32.csv'
path4 = '~/workspace/some_data/intra/results/intra_16.csv'
df1 = pd.read_csv(path1, header=None)
df2 = pd.read_csv(path2, header=None)
df3 = pd.read_csv(path3, header=None)
df4 = pd.read_csv(path4, header=None)
plot_stability_vs_noisiness(df1.values, df2.values, df3.values, df4.values)
"""

"""
# plot frequency distribution
path1 = '~/workspace/some_data/dist/distances_128.csv'
path2 = '~/workspace/some_data/dist/distances_64.csv'
path3 = '~/workspace/some_data/dist/distances_32.csv'
path4 = '~/workspace/some_data/dist/distances_16.csv'
df1 = pd.read_csv(path1, header=None)
df2 = pd.read_csv(path2, header=None)
df3 = pd.read_csv(path3, header=None)
df4 = pd.read_csv(path4, header=None)
plot_frequency_distribution(df1.values, df2.values, df3.values, df4.values)
"""

"""
# plot chain's distribution
path1 = '~/workspace/some_data/logs/various.log'
path2 = '~/workspace/some_data/logs/32p-result.log'
path3 = '~/workspace/some_data/logs/64p-result.log'
path4 = '~/workspace/some_data/logs/128p-result.log'
df1 = pd.read_csv(path1, sep='\t', header=None)
df2 = pd.read_csv(path2, sep='\t', header=None)
df3 = pd.read_csv(path3, sep='\t', header=None)
df4 = pd.read_csv(path4, sep='\t', header=None)

data = np.array(df1.values[:, 11]) # 10 instead of 11 before
lines = len(data)
matrix = np.zeros((lines, 2))
for i, array in enumerate(data):
    a = array.split(',')
    for j, acc in enumerate(a):
        val = float(acc)
        matrix[i, j] = val

#for i, accs in enumerate(matrix):
#    matrix[i, :] = polarize_accuracies(accs)
#for i, accs in enumerate(matrix):
#    matrix[i, :] = process_accuracies(accs)
processed = process_accuracies(matrix)
plot_individual_learning_rates(processed)
"""

#"""
path = '~/workspace/some_data/logs/various.log'
df = pd.read_csv(path, sep='\t', header=None)

lines = np.array(df.values[:, :])

accuracies = np.zeros((len(lines)))
ks = np.zeros((len(lines)))
ns = np.zeros((len(lines)))
noisinesses = np.zeros((len(lines)))
nums = np.zeros((len(lines)))
reps = np.zeros((len(lines)))
pop_sizes = np.zeros((len(lines)))

for i, line in enumerate(lines):
    accuracies[i] = float(line[10])
    ks[i] = float(line[4])
    ns[i] = float(line[3])
    noisinesses[i] = float(line[7])
    nums[i] = float(line[6])
    reps[i] = float(line[8])
    pop_sizes[i] = float(line[9])

#plot_param_influence(accuracies, reps)
#plot_param_cdf(accuracies, reps)

#parameters = np.array([ks, ns, noisinesses, nums, reps, pop_sizes])
#names = ['k', 'n', 'noisiness', 'num', 'reps', 'pop_size']
#plot_params_histograms(parameters, names)

#"""

path = '~/workspace/some_data/logs/various.log'
plot_hyperparams(path)
