import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcdefaults()


def plot_individual_learning_rates(accuracies_sorted):
    _, k = np.shape(accuracies_sorted)
    chains = np.arange(k)
    means = np.mean(accuracies_sorted, axis=0)
    plt.figure(figsize=(20, 15))
    plt.plot(np.swapaxes(accuracies_sorted, 0, 1), 'bo', alpha=1.0 / k)
    mean_line, = plt.plot(means, 'k:', label='means')
    plt.legend(handles=[mean_line], loc=1, fontsize=20)
    plt.xticks(chains, chains)
    plt.xlabel('chains', fontsize=20)
    plt.ylabel('accuracy mean', fontsize=20)
    plt.title('learning rates of individual chains', fontsize=20)
    plt.grid(True)
    plt.show()


def polarize_accuracies(accs):
    for i, accuracy in enumerate(accs):
        if accuracy < 0.5:
            accs[i] = 1.0 - accuracy
    return accs


def process_accuracies(accs):
    n, k = np.shape(accs)
    for i in range(n):
        accs[i, :] = np.sort(polarize_accuracies(accs[i, :]))[::-1]
    return accs


def plot_frequency_distribution(d1, d2, d3, d4):
    labels = ['(128x1)-PUF', '(64x1)-PUF', '(32x1)-PUF', '(16x1)-PUF']
    data = np.concatenate((d1, d2, d3, d4), axis=1)
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.hist(data, 100, histtype='bar', label=labels, stacked=True)
    plt.xticks(np.arange(0.5, 0.8, 0.05))
    plt.xlabel('distance', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    plt.title('frequency distribution of distances', fontsize=20)
    legend = plt.legend(loc='center', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('large')
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.axis([0.5, 0.8, 0, 2500])
    ax.text(0.71, 2100, 'This diagram results from distances\n'
                        '- where 0.4 equals 0.6 -\n'
                        'of 10,000 times each 2 randomly sampled PUFs\n'
                        'compared by each 100,000 randomly sampled\n'
                        'challenges for lengths 16, 32, 64 and 128.',
            bbox={'facecolor': 'white', 'pad': 10})
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.8)
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color='grey', alpha=0.5)
    plt.show()


def plot_stability_vs_noisiness(d1, d2, d3, d4):
    left = 0
    right = 2
    top = 1.02
    bottom = 0.48
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.plot([left, right], [0.5, 0.5], 'k--', linewidth=3)
    plt.plot([left, right], [1.0, 1.0], 'k--', linewidth=3)
    plt.xticks(np.arange(left, right + 0.1, 0.05), rotation='vertical')
    plt.yticks(np.arange(bottom + 0.02, top, 0.02))
    x = np.arange(0.001, 2.001, 0.001)
    plt.plot(x, d1, linewidth=0.5, label='(128x1)-PUF')
    plt.plot(x, d2, linewidth=0.5, label='(64x1)-PUF')
    plt.plot(x, d3, linewidth=0.5, label='(32x1)-PUF')
    plt.plot(x, d4, linewidth=0.5, label='(16x1)-PUF')
    legend = plt.legend(loc='center left', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('large')
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.xlabel('noisiness', fontsize=20)
    plt.ylabel('stability', fontsize=20)
    plt.title('stability vs. noisiness', fontsize=20)
    plt.axis([left, right, bottom, top])
    ax.text(1.4, 0.9, 'This diagram results from means of reliabilities\n'
                      'of each 1,000 randomly sampled (128x1)-PUFs\n'
                      'evaluated by each 100,000 randomly sampled\n'
                      'challenges for 2,000 different noisinesses\n'
                      'from 0.001 to 2.',
            bbox={'facecolor': 'white', 'pad': 10})
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.8)
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color='grey', alpha=0.5)
    plt.show()


def plot_param_influence(accuracies, param, bins=16):
    plt.subplots(figsize=(20, 15))
    variations = np.unique(param)
    plt.hist2d(accuracies, param, bins=(bins, len(variations)))
    plt.colorbar().set_label('absolute frequency')
    plt.title('2D histogram of accuracies per value of parameter')
    plt.xlabel('accuracy')
    plt.ylabel('value of param')
    plt.show()


def plot_param_cdf(accuracies, param):
    plt.figure(figsize=(20, 15))
    num = len(param)
    nums = range(num)
    variations = np.unique(param)
    for i in variations:
        indices = param[nums] == i
        data = accuracies[indices]
        counts, bin_edges = np.histogram(data, bins=num, normed=True)
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[1:], cdf / cdf[-1])
        plt.title('CDFs of accuracies per value of parameter')
        plt.xlabel('accuracy')
        plt.ylabel('cumulated frequency')
        plt.legend(variations)
    plt.show()


def plot_params_histograms(parameters, names):
    fig = plt.figure(figsize=(20, 15))
    num = len(parameters)
    for i, parameter in enumerate(parameters):
        fig.add_subplot(num, 1, i+1)
        hist = np.histogram(parameter)
        plt.hist(hist)
        plt.ylabel(names[i])
    plt.show()

def group_by_hypervalues(data, param):
    dict = {
        'num': (6, 6),
        'reps': (8, 8),
        'pop_size': (9, 9),
        'num_reps': (6, 8),
        'num_popsize': (6, 9),
        'reps_popsize': (8, 9),
    }
    col = dict[param]
    if col[0] == col[1]:
        col = col[0]
        values = []
        for val in data[:, col]:
            if val not in values:
                values.append(val)
        values.sort()
        res = []
        for val in values:
            temp = []
            for row in range(len(data)):
                if data[row, col] == val:
                    temp.append(data[row, 10])
            res.append(np.array(temp))
        return res, values

    values1 = []
    values2 = []
    for val in data[:, col[0]]:
        if val not in values1:
            values1.append(val)
    values1.sort()
    for val in data[:, col[1]]:
        if val not in values2:
            values2.append(val)
    values2.sort()
    res = []
    for val1 in values1:
        for val2 in values2:
            temp = []
            for row in range(len(data)):
                if data[row, col[0]] == val1 and data[row, col[1]] == val2:
                    temp.append(data[row, 10])
            res.append(np.array(temp))
    tuples = []
    for i, val1 in enumerate(values1):
        if val1 >= 1000:
            values1[i] = str(int(val1 / 1000)) + 'k'
    for j, val2 in enumerate(values2):
        if val2 >= 1000:
            values1[j] = str(int(val2 / 1000)) + 'k'
    for i in range(len(values1)):
        for j in range(len(values2)):
            tuples.append((values1[i], values2[j]))
    return res, tuples

def plot_hyperparams(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df = np.array(df)
    n = df[0, 3]
    k = df[0, 4]
    noisiness = str(df[0, 7])

    params = ['num', 'reps', 'pop_size', 'num_reps', 'num_popsize', 'reps_popsize']
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 16))
    plt.suptitle('Overview of adequacy of hyperparameter values\n'
                 'from learning (%i, %i)-XOR-Arbiter-PUFs with noisiness %s'
                 % (n, k, noisiness), fontsize=24)

    data = []
    labels = []
    for param in params:
        d, l = group_by_hypervalues(df, param)
        data.append(d)
        labels.append(l)

    boxes = []

    for i in range(2):
        for j in range(3):
            p = i * 3 + j
            boxes.append(axes[i, j].boxplot(data[p], labels=labels[p],
                                            patch_artist=True, showmeans=True, meanline=True))
            if len(np.shape(labels[p][0])) > 0:
                for cnt, l in enumerate(labels[p]):
                    labels[p][cnt] = str(l).replace("\'", "").replace("(", "").replace(")", "")
                axes[i, j].set_xticklabels(labels=labels[p], rotation=45)
            axes[i, j].set_xlabel(params[p], fontsize=16)
            means = []
            num_boxes = np.shape(data[p])[0]
            for box in range(num_boxes):
                if not list(data[p][box]):
                    means.append(0.5)
                else:
                    means.append(np.mean(data[p][box]))
            m = max(means)
            best = [i for i, j in enumerate(means) if j == m]
            colors = num_boxes * ['w']
            for index in best:
                colors[index] = 'greenyellow'
            for patch, color in zip(boxes[p]['boxes'], colors):
                patch.set_facecolor(color)
            start = means[0]
            end = means[-1]
            mean_line, = axes[i, j].plot([start] + means + [end], 'g', label='means')
            axes[i, j].legend(handles=[mean_line])

    fig.text(0.08, 0.3, 'accuracies', ha='center', rotation='vertical', fontsize=16)
    fig.text(0.08, 0.75, 'accuracies', ha='center', rotation='vertical', fontsize=16)
    plt.show()
