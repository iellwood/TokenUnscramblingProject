import numpy as np
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import scipy.stats as stats


# Define a custom version of imshow for consistency across all the plots.
def imshow(plot_axis, matrix):
    """
    A customized version of imshow for the vec_dim averaged scramble matrices.
    A value of zero is set to black. White is set to twice the mean of the scramble matrix.
    Hence a constant scramble matrix will appear as pure gray (0.5, 0.5, 0.5).
    :param plot_axis: The matplotlib axis to plot on
    :param matrix: The vec_dim averaged scramble matrix
    """
    plot_axis.imshow(matrix, cmap='gray', vmin=0, vmax= 2 *np.mean(matrix))

def plot_nll_data(path):
    """
    Plots the test loss as a function of training epoch
    :param path: Path to the file nlls.npz created by GetTestNLLs.py
    """
    data_dict = np.load(path)



    nlls_vanilla = data_dict['nlls_vanilla_transformer']
    nlls_modified_model = data_dict['nlls_modified_model']


    # for i in range(nlls_vanilla.shape[0]):
    #     plt.scatter(np.array([10, 20, 30, 40]) - 1, nlls_vanilla[i, 1:], color='k')
    #
    # for i in range(nlls_modified_model.shape[0]):
    #     plt.scatter(np.array([10, 20, 30, 40]) + 1, nlls_modified_model[i, 1:], color=prettyplot.colors['blue'])

    mean_vanilla = np.mean(nlls_vanilla, axis=0)
    sem_vanilla = np.std(nlls_vanilla, axis=0)/np.sqrt(nlls_vanilla.shape[0])

    mean_modified = np.mean(nlls_modified_model, axis=0)
    sem_modified = np.std(nlls_modified_model, axis=0)/np.sqrt(nlls_modified_model.shape[0])

    difference_mean = (mean_modified - mean_vanilla)/mean_vanilla * 100
    difference_sem = (np.sqrt(sem_vanilla**2 + sem_modified**2))/mean_vanilla * 100

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    plt.errorbar(
        x=np.array([0, 10, 20, 30, 40]) - 1,
        y=mean_vanilla,
        yerr=sem_vanilla,
        color='k', capsize=5, marker='o',
        linestyle='',
        label='Vanilla'
    )

    plt.errorbar(
        x=np.array([0, 10, 20, 30, 40])+ 2,
        y=mean_modified,
        yerr=sem_modified,
        color=prettyplot.colors['blue'], capsize=5, marker='o',
        linestyle='',
        label='Modified'
    )

    prettyplot.no_box()
    prettyplot.xlabel('Epoch')
    prettyplot.ylabel('NLL')
    prettyplot.title('NLL vs. Epoch')
    plt.ylim([0, 2000])
    plt.savefig('Plots/NLLVsEpoch.pdf')
    plt.show()

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)


    plt.errorbar(
        x=[0, 10, 20, 30, 40],
        y=difference_mean,
        yerr=difference_sem,
        color='k', capsize=5, marker='o',
        label='Difference'
    )

    for i in range(5):
        s = stats.ttest_ind(nlls_vanilla[:, i], nlls_modified_model[:, i])
        if s.pvalue < 0.05:
            plt.text(i * 10 - 5, difference_mean[i] * 1.4, "p = " + str(s.pvalue))
            print('Significant different in test loss. p =', s.pvalue, s)

    prettyplot.no_box()

    prettyplot.xlabel('Epoch')
    prettyplot.ylabel('% Change in NLL')
    prettyplot.title('% Change in NLL vs. Epoch')
    plt.savefig('Plots/PercentChangeInNLLVsEpoch.pdf')
    plt.show()

def sparsity_metric(matrix, sparsity_axis):
    """
    The sparsity metric used for the rows of the scramble matrices
    :param matrix: A vec_dim averaged sparsity matrix
    :param sparsity_axis: The axis along which to perform the sparsity metric
    :return: sparsity
    """
    n = matrix.shape[sparsity_axis]
    L_1 = np.sum(np.abs(matrix), sparsity_axis)
    L_2 = np.sqrt(np.sum(np.square(matrix), sparsity_axis)) + 1e-10
    return np.mean((L_1/L_2 - 1)/(np.sqrt(n) - 1))


def sort_scramble_matrices_by_sparsity(matrices, sparsity_axis, PCA_dimension=10):
    """
    Uses the sparsity metric to sort the sparsity matrices.
    NOTE: The
    :param matrices: The input key scramble matrices
    :param sparsity_axis: Which axis to apply the sparsity metric to.
    :PCA_dimension: The amount of PCA_reduction to use before sorting. Default is 10 dimensions
    :return: sorted_matrices, permutation
    """
    if len(matrices.shape) == 4:
        I = np.argsort(np.array([sparsity_metric(matrices[i, -1, :, :], sparsity_axis) for i in range(matrices.shape[0])]))
        return matrices[I, :, :, :], I
    elif len(matrices.shape) == 5:
        I = np.argsort(np.array([sparsity_metric(matrices[PCA_dimension - 1, i, -1, :, :], sparsity_axis) for i in range(matrices.shape[1])]))
        return matrices[:, I, :, :, :], I

def plot_perimaximum_average(values_or_keys,
                             M_K, M_V, seq_len,
                             number_of_PCA_components_index,
                             checkpoint_index,
                             models_to_include=None,
                             color='k',
                             label=None,
                             window_width=9):
    if models_to_include is None:
        models_to_include = np.arange(M_K.shape[1])
    if values_or_keys == 'keys':
        M_proj_avg_pca = M_K[number_of_PCA_components_index, :, checkpoint_index, :, :]
    else:
        M_proj_avg_pca = M_V[number_of_PCA_components_index, :, checkpoint_index, :, :]
    peri_maximum_average = np.zeros([len(models_to_include), seq_len, 2 * window_width + 1])
    peri_maximum_counts = np.zeros([len(models_to_include), seq_len, 2 * window_width + 1])
    for model_average_index, model_index in enumerate(models_to_include):
        for j in range(seq_len):
            v = M_proj_avg_pca[model_index, j, :]
            max_index = np.argmax(v)
            for k in range(-window_width, window_width + 1):
                index = max_index + k
                if 0 <= index < seq_len:
                    peri_maximum_average[model_average_index, j, k + window_width] = v[index]
                    peri_maximum_counts[model_average_index, j, k + window_width] += 1
    peri_maximum_average = np.sum(peri_maximum_average, 1)/np.sum(peri_maximum_counts, 1)

    mean = np.nanmean(peri_maximum_average, axis=0)
    sem = np.std(peri_maximum_average, axis=0) / np.sqrt(peri_maximum_average.shape[0])

    # Normalize by the maximum
    mx = np.max(mean)
    mean = mean / mx
    sem = sem / mx


    plt.plot(
        np.arange(2 * window_width + 1) - window_width,
        mean,
        color=color,  marker='',
        label=label)