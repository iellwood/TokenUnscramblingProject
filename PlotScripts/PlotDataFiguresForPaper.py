"""
This script produces the complete set of plots for the paper

"Token unscrambling in fixed-weight biological models of transformer attention", I.T. Ellwood 2026.

and saves them in the folder <Plots/> which must exist for the script to run.

This code requires the output of the scripts

    CollectDataFromCheckpoint.py
    GetTestNLLs.py

which are saved in files,

    ModelData/nlls.npz
    ModelData/CheckpointAnalysisResults.obj

**WARNING**: If you rerun CollectDataFromCheckpoint, it will change the results in small ways as it will collect new random
samples for the various plots and statistical tests.

We note that the PCA analysis performed for the keys and queries in the paper is also performed for the values in this
script. We did not include these figures as we did not feel that there was much motivation for the PCA reduction for
the value scramble matrices. For curiosity's sake, we have included the analysis here. Simply set

    plot_PCA_analysis_for_value_scramble_matrix = True

And the plots for the PCA analysis of the values will be generated.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import utils.prettyplot as prettyplot
import scipy.stats as stats
import HelperFunctionsForPlots

# PATHS TO ESSENTIAL DATA FILES:

# 1) Path to the file nlls.npz needed for plotting the test loss
nll_data_path = '../ModelData/nlls.npz'

# 2) Path to the file CheckpointAnalysisResults.obj needed for all other plots.
scramble_matrix_data_path = '../ModelData/CheckpointAnalysisResults.obj'

# Set the batch size for pytorch evaluations.
batch_size = 32

# Change the variable plot_PCA_analysis_for_value_scramble_matrix to true if you would like to see the effect of PCA reduction on the
# value scramble matrices. Our general impression is that there is no benefit to this PCA reduction, perhaps because the
# output of the scrambled attention layer depends linearly on the value scramble matrix, making it train quickly.
plot_PCA_analysis_for_value_scramble_matrix = False

# Plots of the test loss for the scrambled model during training, compared with a vanilla transformer.
plot_test_loss_for_scrambled_and_unscrambled_models = False
if plot_test_loss_for_scrambled_and_unscrambled_models:
    HelperFunctionsForPlots.plot_nll_data(nll_data_path)

"""
Load the data collection by CollectDataFromCheckpoint and unpack the dictionary into individual variables.
"""
with open(scramble_matrix_data_path, 'rb') as file:
    scramble_matrices_dict = pickle.load(file)
scramble_matrices_K_token_average = scramble_matrices_dict['M_K_proj_token_average']
scramble_matrices_V_token_average = scramble_matrices_dict['M_V_proj_token_average']
softmax_weights = scramble_matrices_dict['softmax_w']
checkpoints = scramble_matrices_dict['checkpoints']
number_of_models = scramble_matrices_dict['number_of_models']
number_of_checkpoints = scramble_matrices_dict['number_of_checkpoints']
PCA_dimensions = scramble_matrices_dict['PCA_dimensions']
seq_len = scramble_matrices_dict['seq_len']
vec_dim = scramble_matrices_dict['vec_dim']
v_tilde_norms_list = scramble_matrices_dict['v_tilde_norms_list']
max_w_indices_list = scramble_matrices_dict['max_w_indices_list']
repetitions = scramble_matrices_dict['repetitions']
permutation_invariance_bit_mask = scramble_matrices_dict['permutation_invariance_bit_mask']
explained_variance_q = scramble_matrices_dict['explained variance by PCA dimension, query']
explained_variance_k = scramble_matrices_dict['explained variance by PCA dimension, key']
explained_variance_v = scramble_matrices_dict['explained variance by PCA dimension, value']


"""
Sort the scramble matrices by their sparsity. PCA_dimension=10 is used as this dimension has the highest sparsity for the 
key scramble matrices. Note that the only results that depend on this sort are the two plots that separate the 
models into thirds based on their sparsity. Note also the the value scramble matrices are sorted by the same permutation,
"""
M_K, model_permutation = HelperFunctionsForPlots.sort_scramble_matrices_by_sparsity(np.array(scramble_matrices_K_token_average), sparsity_axis=1, PCA_dimension=10)
M_V = np.array(scramble_matrices_V_token_average)
M_V = M_V[:, model_permutation, :, :, :]

plot_explained_variance = False
if plot_explained_variance:
    # Plot The explained variance as a function of PCA reduction dimension

    # Make a figure with correct size and padding
    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Use the cumulative sum of the explained variances per dimension.
    # for the keys, average over the token index.
    explained_variance_q = np.cumsum(explained_variance_q[:, -1, :], axis=1)
    explained_variance_k = np.mean(np.cumsum(explained_variance_k[:, -1, :, :], axis=2), 1)

    # Plots the explained variance as a function of PCA dimension
    plt.plot(np.arange(64) + 1, explained_variance_q[0, :], color=prettyplot.colors['red'], label='q')
    plt.plot(np.arange(64) + 1, explained_variance_k[0, :], color=prettyplot.colors['green'], label='k')

    # To see the explained variance for the values, set plot_explained_variance_for_values = True

    if plot_PCA_analysis_for_value_scramble_matrix:
        explained_variance_v = np.mean(np.cumsum(explained_variance_v[:, -1, :, :], axis=2), 1)
        plt.plot(np.arange(64) + 1, explained_variance_v[0, :], color=prettyplot.colors['blue'], label='v')

    plt.ylim([0, 1.1])
    prettyplot.no_box()
    prettyplot.title('fraction EV')
    prettyplot.xlabel('PCA dimensions')
    plt.legend(frameon=False, loc='lower right')
    plt.xticks([1, 2, 4, 8, 16, 32, 64])
    plt.savefig('Plots/Explained_variance_vs_PCA_dimension.pdf')
    plt.show()

# Plots of the vec_dim averaged scramble matrices
plot_images_of_scramble_matrices = False
if plot_images_of_scramble_matrices:
    # These are the PCA_dimensions shown in the plot as illustrations of the effect of PCA dimension reduction
    Example_PCA_Dimensions = [1, 2, 4, 8, 16, 32, 64]
    checkpoints_to_plot = [0, 1, 4]  # Only show the untrained model and model trained for 10 and 40 epochs

    fig, axes = plt.subplots(len(checkpoints_to_plot), number_of_models, figsize=(12, 2.25))
    M_no_PCA = M_K[-1]
    for i in range(len(checkpoints_to_plot)):
        for j in range(number_of_models):
            axis = axes[i][j]
            axis.axis('off')
            HelperFunctionsForPlots.imshow(axis, M_no_PCA[j, checkpoints_to_plot[i]])
    prettyplot.subplots_title(fig, 'Key scramble matrices by epoch (no PCA)')
    plt.savefig('Plots/ScrambleMatricesByEpoch_K.pdf')
    plt.show()

    # EFFECT OF EPOCH ON VALUE SCRAMBLE MATRICES
    fig, axes = plt.subplots(len(checkpoints_to_plot), number_of_models, figsize=(12, 2.25))
    M_V_no_PCA = M_V[-1]
    for i in range(len(checkpoints_to_plot)):
        for j in range(number_of_models):
            axis = axes[i][j]
            axis.axis('off')
            HelperFunctionsForPlots.imshow(axis, M_V_no_PCA[j, checkpoints_to_plot[i]])
    prettyplot.subplots_title(fig, 'Value scramble matrices by epoch (no PCA)')
    plt.savefig('Plots/ScrambleMatricesByEpoch_V.pdf')
    plt.show()

    # EFFECT OF PCA ON KEY SCRAMBLE MATRICES
    M_K_proj_avg = M_K
    fig, axes = plt.subplots(len(Example_PCA_Dimensions), number_of_models, figsize=(12, 5.5))
    for pca_index in range(len(Example_PCA_Dimensions)):
        for model_index in range(M_K_proj_avg.shape[1]):
            axis = axes[pca_index][model_index]
            axis.axis('off')
            HelperFunctionsForPlots.imshow(axis, M_K_proj_avg[Example_PCA_Dimensions[pca_index] - 1, model_index, -1])
    prettyplot.subplots_title(fig, 'Effect of PCA dim on Key scramble matrices: ' + str(Example_PCA_Dimensions))
    plt.savefig('Plots/EffectOfPCAOnScrambleMatrices_K.pdf')
    plt.show()

    if plot_PCA_analysis_for_value_scramble_matrix:
        # EFFECT OF PCA ON VALUE SCRAMBLE MATRICES
        M_V_proj_avg = M_V
        fig, axes = plt.subplots(len(Example_PCA_Dimensions), number_of_models, figsize=(12, 5.5))
        for pca_index in range(len(Example_PCA_Dimensions)):
            for model_index in range(M_V_proj_avg.shape[1]):
                axis = axes[pca_index][model_index]
                axis.axis('off')
                HelperFunctionsForPlots.imshow(axis, M_V_proj_avg[Example_PCA_Dimensions[pca_index] - 1, model_index, -1])
        prettyplot.subplots_title(fig, 'Effect of PCA dim Value scramble matrices: ' + str(Example_PCA_Dimensions))
        plt.savefig('Plots/EffectOfPCAOnScrambleMatrices_V.pdf')
        plt.show()

# sums = np.zeros((15, 7))
# counts = np.zeros((15, 7))
#
# window_width = 9
# models_to_include_list = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
# colors = [prettyplot.colors['red'], prettyplot.colors['green'], prettyplot.colors['blue']]
# pca_component = 1

plot_perimaximum_averages = False
if plot_perimaximum_averages:

    Example_PCA_Dimensions = [8, 16, 32, 64]

    # PERIMAXIMUM AVERAGES VS EPOCH (KEYS)


    colors_for_plot = [prettyplot.colors['blue'], prettyplot.colors['green'], prettyplot.colors['red'], 'k']

    number_of_PCA_components_index = -1




    # Perimaximum average vs. Epoch KEYS

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    for checkpoint_index in range(number_of_checkpoints):
        HelperFunctionsForPlots.plot_perimaximum_average('keys',
                                                         M_K, M_V, seq_len,
                                                         -1,
                                                         checkpoint_index,
                                                         color=prettyplot.color_list[checkpoint_index],
                                                         label= [0, 10, 20, 30, 40][checkpoint_index])

    plt.ylim([0, 1.1])
    prettyplot.no_box()
    prettyplot.title('perimaximum average vs Epoch (K)')
    prettyplot.xlabel('token distance from max')
    plt.legend(frameon=False, loc='lower left')
    plt.savefig('Plots/PerimaximumAverageOfScrambleMatrixRow_vs_Epoch_K.pdf')
    plt.show()

    # Perimaximum average vs. Epoch VALUES

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    for checkpoint_index in range(number_of_checkpoints):
        HelperFunctionsForPlots.plot_perimaximum_average('values',
                                                         M_K, M_V, seq_len,
                                                         -1, checkpoint_index,
                                                         color=prettyplot.color_list[checkpoint_index],
                                                         label= [0, 10, 20, 30, 40][checkpoint_index])

    plt.ylim([0, 1.1])
    prettyplot.no_box()
    prettyplot.title('perimaximum average vs Epoch (V)')
    prettyplot.xlabel('token distance from max')
    plt.legend(frameon=False)
    plt.savefig('Plots/PerimaximumAverageOfScrambleMatrixRow_vs_Epoch_V.pdf')

    plt.show()


    # PERIMAXIMUM AVERAGES FOR THE K SCRAMBLE MATRIX ROWS VS PCA DIM

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    for PCA_dimension_index in range(len(Example_PCA_Dimensions)):
        HelperFunctionsForPlots.plot_perimaximum_average('keys',
                                                         M_K, M_V, seq_len,
                                                         Example_PCA_Dimensions[PCA_dimension_index] - 1, -1,
                                                         color=prettyplot.color_list[PCA_dimension_index],
                                                         label=Example_PCA_Dimensions[PCA_dimension_index])
    plt.ylim([0, 1.1])
    prettyplot.no_box()
    prettyplot.title('perimaximum average vs PCA (K)')
    prettyplot.xlabel('token distance from max')
    plt.legend(frameon=False)
    plt.savefig('Plots/PerimaximumAverageOfScrambleMatrixRow_vs_PCA_K.pdf')

    plt.show()

    if plot_PCA_analysis_for_value_scramble_matrix:
        # PERIMAXIMUM AVERAGES FOR THE V SCRAMBLE MATRIX ROWS VS PCA DIM

        fig = plt.figure(figsize=(3, 3))
        plt.subplots_adjust(left=0.25, bottom=0.25)
        for PCA_dimension_index in range(len(Example_PCA_Dimensions)):
            HelperFunctionsForPlots.plot_perimaximum_average('values',
                                                             M_K, M_V, seq_len,
                                                             Example_PCA_Dimensions[PCA_dimension_index] - 1, -1,
                                                             color=prettyplot.color_list[PCA_dimension_index],
                                                             label=Example_PCA_Dimensions[PCA_dimension_index])
        plt.ylim([0, 1.1])
        prettyplot.no_box()
        prettyplot.title('perimaximum average vs PCA (V)')
        prettyplot.xlabel('token distance from max')
        plt.legend(frameon=False)
        plt.savefig('Plots/PerimaximumAverageOfScrambleMatrixRow_vs_PCA_V.pdf')
        plt.show()


    # Perimaximum average of the scramble matrix rows for the top 5, middle 5 and worst 5 models (Key scramble matrix)
    # We selected 10 dimensions for the PCA reduction as this produced the largest sparsity for the best models.

    models_to_include_list = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    PCA_dim = 10
    HelperFunctionsForPlots.plot_perimaximum_average('keys',
                             M_K, M_V, seq_len,
                             PCA_dim - 1, -1,
                             models_to_include=models_to_include_list[0],
                             color=prettyplot.colors['green'],
                             label='top 5')

    HelperFunctionsForPlots.plot_perimaximum_average('keys',
                             M_K, M_V, seq_len,
                             PCA_dim - 1, -1,
                             models_to_include=models_to_include_list[1],
                             color=prettyplot.colors['blue'],
                             label='Mid 5')

    HelperFunctionsForPlots.plot_perimaximum_average('keys',
                             M_K, M_V, seq_len,
                             PCA_dim - 1, -1,
                             models_to_include=models_to_include_list[2],
                             color=prettyplot.colors['red'],
                             label='bot 5')

    HelperFunctionsForPlots.plot_perimaximum_average('keys',
                             M_K, M_V, seq_len,
                             PCA_dim - 1, -1,
                             models_to_include=np.arange(number_of_models),
                             color=prettyplot.colors['black'],
                             label='all')

    plt.ylim([0, 1.1])
    prettyplot.no_box()
    prettyplot.title(f'perimaximum average PCA{PCA_dim} (K)')
    prettyplot.xlabel('token distance from max')
    plt.legend(frameon=False)
    plt.savefig(f'Plots/PerimaximumAverageOfScrambleMatrixRow_PCA{PCA_dim}_K.pdf')
    plt.show()

    # Perimaximum average of the scramble matrix rows for the top 5, middle 5 and worst 5 models (Value scramble matrix)
    # Note that for the value scramble matrix, we have not performed and PCA reduction as we did not feel that there
    # was motivation to do so.

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    PCA_dim = 64
    HelperFunctionsForPlots.plot_perimaximum_average('values',
                                                     M_K, M_V, seq_len,
                                                     PCA_dim - 1, -1,
                                                     models_to_include=models_to_include_list[0],
                                                     color=prettyplot.colors['green'],
                                                     label='top 5')

    HelperFunctionsForPlots.plot_perimaximum_average('values',
                                                     M_K, M_V, seq_len,
                                                     PCA_dim - 1, -1,
                                                     models_to_include=models_to_include_list[1],
                                                     color=prettyplot.colors['blue'],
                                                     label='Mid 5')

    HelperFunctionsForPlots.plot_perimaximum_average('values',
                                                     M_K, M_V, seq_len,
                                                     PCA_dim - 1, -1,
                                                     models_to_include=models_to_include_list[2],
                                                     color=prettyplot.colors['red'],
                                                     label='bot 5')

    HelperFunctionsForPlots.plot_perimaximum_average('values',
                                                     M_K, M_V, seq_len,
                                                     PCA_dim - 1, -1,
                                                     models_to_include=np.arange(number_of_models),
                                                     color=prettyplot.colors['black'],
                                                     label='all')

    plt.ylim([0, 1.1])
    prettyplot.no_box()
    prettyplot.title('perimaximum average noPCA (V)')
    prettyplot.xlabel('token distance from max')
    plt.legend(frameon=False)
    plt.savefig(f'Plots/PerimaximumAverageOfScrambleMatrixRow_PCA{PCA_dim}_V.pdf')
    plt.show()

# Plots of the sparsity metric for keys and values during training and vs. PCA reduction dimension
plot_sparsity_metric_plots = False
if plot_sparsity_metric_plots:

    # Example_PCA_Dimensions are the PCA reduction dimensions  used for example plots.
    # Note that above we included 1, 2, 4 as well, but did not include them
    # here as these dimensions appear to only have high sparsity due to noisiness.
    Example_PCA_Dimensions = [8, 16, 32, 64]

    # Computes the sparsity metric for the key scramble matrix rows
    colors_for_plot = [prettyplot.colors['blue'], prettyplot.colors['green'], prettyplot.colors['red'], 'k']
    m_k = M_K
    m_shape_old = m_k.shape
    m_k = np.reshape(m_k, [-1, 20, 20])
    sparsity_metrics_k = np.zeros((m_k.shape[0],))
    for i in range(m_k.shape[0]):
        sparsity_metrics_k[i] = HelperFunctionsForPlots.sparsity_metric(m_k[i, :, :], 1)
    sparsity_metrics_k = 1 - sparsity_metrics_k  # compute sparsity instead of the sparsity loss
    m_k = np.reshape(m_k, m_shape_old)
    sparsity_metrics_k = np.reshape(sparsity_metrics_k, m_shape_old[:-2])  # shape = [number_of_pca_dimensions, number_of_models, number_of_checkpoints]
    sparsity_mean = np.mean(sparsity_metrics_k, 1)
    sparsity_sem = np.std(sparsity_metrics_k, 1)/np.sqrt(sparsity_metrics_k.shape[1])

    # Computes the sparsity metric for the value scramble matrix rows
    m_v = M_V
    m_v_shape_old = m_v.shape
    m_v = np.reshape(m_v, [-1, 20, 20])
    sparsity_metrics_v = np.zeros((m_v.shape[0],))
    for i in range(m_v.shape[0]):
        sparsity_metrics_v[i] = HelperFunctionsForPlots.sparsity_metric(m_v[i, :, :], 1)
    sparsity_metrics_v = 1 - sparsity_metrics_v  # compute sparsity instead of the sparsity loss
    m_v = np.reshape(m_v, m_v_shape_old)
    sparsity_metrics_v = np.reshape(sparsity_metrics_v, m_v_shape_old[:-2])  # shape = [number_of_pca_dimensions, number_of_models, number_of_checkpoints]
    sparsity_mean_v = np.mean(sparsity_metrics_v, 1)
    sparsity_sem_v = np.std(sparsity_metrics_v, 1) / np.sqrt(sparsity_metrics_v.shape[1])

    # PLOT: Sparsity as a function of epoch. NO PCA (KEY SCRAMBLE MATRIX)

    # Make a figure with correct size and padding
    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Plots the results for individual models
    for i in range(number_of_models):
        plt.plot(np.arange(number_of_checkpoints) * 10, sparsity_metrics_k[-1, i, :], color=[0.8, 0.8, 0.8])

    # Plot the average sparsity across models with sem.
    plt.errorbar(
        x=np.arange(number_of_checkpoints) * 10,
        y=sparsity_mean[-1, :],
        yerr=sparsity_sem[-1, :],
        color='k', capsize=5, marker='o', label='sparsity')
    prettyplot.no_box()
    prettyplot.xlabel('training epochs')
    prettyplot.ylabel('sparsity')
    plt.xticks(np.arange(number_of_checkpoints) * 10)
    prettyplot.title('Sparsity vs. Epoch. NO PCA')
    plt.savefig('Plots/SparsityVsEpochNoPCA.pdf')

    s = stats.ttest_rel(sparsity_metrics_k[-1, :, 0], sparsity_metrics_k[-1, :, -1])
    if s.pvalue < 0.05:
        print(f"K Scramble matrix: Mean sparsity before and after training, p = {s.pvalue}")
        print(f"\tComplete stats {s}")
        print(f"\tMean before = {sparsity_mean[-1, 0]}, mean after = {sparsity_mean[-1, -1]}.")

    plt.show()

    # PLOT: Sparsity as a function of epoch. NO PCA (VALUE SCRAMBLE MATRIX)

    # Make a figure with correct size and padding
    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Plots the results for individual models
    for i in range(number_of_models):
        plt.plot(np.arange(number_of_checkpoints) * 10, sparsity_metrics_v[-1, i, :], color=[0.8, 0.8, 0.8])

    # Plot the average sparsity across models with sem.
    plt.errorbar(
        x=np.arange(number_of_checkpoints) * 10,
        y=sparsity_mean_v[-1, :],
        yerr=sparsity_sem_v[-1, :],
        color='k', capsize=5, marker='o', label='sparsity')
    prettyplot.no_box()
    prettyplot.xlabel('training epochs')
    prettyplot.ylabel('sparsity')
    plt.xticks(np.arange(number_of_checkpoints) * 10)
    prettyplot.title('Spars. vs. Epoch. NO PCA. VALUES')
    plt.savefig('Plots/SparsityVsEpochNoPCA_V.pdf')
    s = stats.ttest_rel(sparsity_metrics_v[-1, :, 0], sparsity_metrics_v[-1, :, -1])
    if s.pvalue < 0.05:
        print(f"V Scramble matrix: Mean sparsity before and after training, p = {s.pvalue}")
        print(f"\tComplete stats {s}")
        print(f"\tMean before = {sparsity_mean_v[-1, 0]}, mean after = {sparsity_mean_v[-1, -1]}.")
    plt.show()

    # PLOT: Sparsity as a function of epoch with PCA Reduction (KEY SCRAMBLE MATRIX)

    PCA_dim = 10  # This dimension gives the peak sparsity for the 5 most sparse models

    # Make a figure with correct size and padding
    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Plots the results for individual models
    for i in range(number_of_models):
        plt.plot(np.arange(number_of_checkpoints) * 10, sparsity_metrics_k[PCA_dim - 1, i, :], color=[0.8, 0.8, 0.8])

    # Plot the average sparsity across models with sem.
    plt.errorbar(
        x=np.arange(number_of_checkpoints) * 10,
        y=sparsity_mean[PCA_dim - 1, :],
        yerr=sparsity_sem[PCA_dim - 1, :],
        color='k', capsize=5, marker='o', label='sparsity')
    prettyplot.no_box()
    prettyplot.xlabel('training epochs')
    prettyplot.ylabel('sparsity')
    plt.xticks(np.arange(number_of_checkpoints) * 10)
    prettyplot.title(f'Sparsity vs. Epoch. {PCA_dim} PCA dims (K)')
    plt.savefig(f'Plots/SparsityVsEpoch{PCA_dim}PCA_K.pdf')
    plt.show()

    if plot_PCA_analysis_for_value_scramble_matrix:
        # PLOT: Sparsity as a function of epoch with PCA Reduction (VALUE SCRAMBLE MATRIX)
        # NOT INCLUDED IN PAPER. There doesn't seem to be much benefit to using PCA reduction on the value scramble
        # matrices, but the code is included here for curiosity's sake.
        PCA_dim = 16  # Set to the desired amount of PCA reduction

        fig = plt.figure(figsize=(3, 3))
        plt.subplots_adjust(left=0.25, bottom=0.25)
        for i in range(number_of_models):
            plt.plot(np.arange(number_of_checkpoints) * 10, sparsity_metrics_v[PCA_dim - 1, i, :], color=[0.8, 0.8, 0.8])

        plt.errorbar(
            x=np.arange(number_of_checkpoints) * 10,
            y=sparsity_mean_v[PCA_dim - 1, :],
            yerr=sparsity_sem_v[PCA_dim - 1, :],
            color='k', capsize=5, marker='o', label='sparsity')
        prettyplot.no_box()
        prettyplot.xlabel('training epochs')
        prettyplot.ylabel('sparsity')
        plt.xticks(np.arange(number_of_checkpoints) * 10)
        plt.ylim([0, np.max(sparsity_metrics_v[PCA_dim - 1, :, :]) * 1.1])
        prettyplot.title(f'Sparsity vs. Epoch. {PCA_dim} PCA dims (V)')
        plt.savefig(f'Plots/SparsityVsEpoch{PCA_dim}PCA_V.pdf')
        plt.show()

    # PLOT: Sparsity as a function of PCA Dimension (KEY SCRAMBLE MATRIX)
    # Plot of sparsity vs. PCA reduction dimension. We noted that this plot was not very interesting when
    # all models were included, simply increasing with decreasing dimension. However, when only the most sparse
    # models are included

    # Make a figure with correct size and padding
    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Builds separate lists for the 5 best, 5 average and 5 worst models ranked by row sparsity of the key
    # scramble matrix.
    models_to_include_list = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

    # Plots sparsity vs. PCA reduction for each of the three groups and for the average over all models.
    for i in range(len(models_to_include_list) + 1):
        if i < len(models_to_include_list):
            models_to_include = models_to_include_list[i]
        else:
            models_to_include = np.arange(number_of_models)

        sparsity_mean = np.mean(sparsity_metrics_k[:, models_to_include, -1], axis=1)
        sparsity_std = np.std(sparsity_metrics_k[:, models_to_include, -1], axis=1)/np.sqrt(len(models_to_include))

        plt.plot(
            PCA_dimensions,
            sparsity_mean,
            #yerr=sparsity_std,
            color=colors_for_plot[i],
            #capsize=5,
            marker='', label=['top third', 'middle third', 'bottom third', 'all'][i])

    prettyplot.no_box()
    prettyplot.xlabel('PCA dimension')
    prettyplot.ylabel('sparsity')
    plt.legend(frameon=False)
    plt.xticks([4] + Example_PCA_Dimensions)
    plt.xlim([4, 64])
    plt.ylim([0, 0.04])
    plt.savefig('Plots/SparsityVsPCADimension_K.pdf')
    plt.show()

    if plot_PCA_analysis_for_value_scramble_matrix:
        # PLOT: Sparsity as a function of PCA Dimension (VALUES). This plot is not included in the paper as we did
        # not feel that there was any benefit to the PCA reduction for the value scramble matrices. The code for the
        # plots is included here for curiosity.

        # Make a figure with correct size and padding
        fig = plt.figure(figsize=(3, 3))
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Builds separate lists for the 5 best, 5 average and 5 worst models ranked by row sparsity of the key
        # scramble matrix.
        models_to_include_list = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

        for i in range(4):
            if i < len(models_to_include_list):
                models_to_include = models_to_include_list[i]
            else:
                models_to_include = np.arange(number_of_models)

            sparsity_mean = np.mean(sparsity_metrics_v[:, models_to_include, -1], axis=1)
            sparsity_std = np.std(sparsity_metrics_v[:, models_to_include, -1], axis=1)/np.sqrt(len(models_to_include))

            plt.errorbar(
                x=PCA_dimensions,
                y=sparsity_mean,
                yerr=sparsity_std,
                color=colors_for_plot[i], capsize=5, marker='.', label=['top third', 'middle third', 'bottom third', 'all'][i])

        prettyplot.no_box()
        prettyplot.xlabel('PCA dimension')
        prettyplot.ylabel('sparsity')
        plt.legend(frameon=False)
        plt.xticks(PCA_dimensions, PCA_dimensions)
        plt.savefig('Plots/SparsityVsPCADimension_V.pdf')
        plt.show()

plot_softmax_weights = False
if plot_softmax_weights:
    # Plots the average of the sorted softmax weights.

    ws = np.reshape(softmax_weights, [number_of_models, number_of_checkpoints, -1, 20])
    ws = np.mean(ws, axis=2)
    for i in range(20):
        s = stats.ttest_rel(ws[:, 0, i], ws[:, -1, i])
        if s.pvalue < 0.05:
            print(f'Difference between w_{20 - i} before and after training: p =', s.pvalue, s)
            print(f'\tMean before = {np.mean(ws[:, 0, i])}. Mean after = {np.mean(ws[:, -1, i])}')

    w_mean = np.mean(ws, axis=0)
    w_sem = np.std(ws, axis=0)/np.sqrt(ws.shape[0])

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(bottom=0.25, left=0.25)

    plt.errorbar(
        x=np.flip(np.arange(20) + 1),
        y=w_mean[0, :],
        yerr=w_sem[0, :],
        color='k', capsize=5, marker='o', label='0 Epochs')

    plt.errorbar(
        x=np.flip(np.arange(20) + 1),
        y=w_mean[-1, :],
        yerr=w_sem[-1, :],
        color=prettyplot.colors['blue'], capsize=5, marker='o', label='40 Epochs')

    plt.xlim([0, 10])
    plt.xticks([1, 5, 10])
    plt.ylim([0, 1])
    plt.yticks([0, 0.5, 1])
    prettyplot.no_box()
    prettyplot.xlabel('rank')
    prettyplot.ylabel('weight')
    plt.legend(frameon=False)
    plt.savefig('Plots/SortedSoftmaxWeightsVsTraining.pdf')
    plt.show()

plot_contribution_of_v_i_to_y = False
if plot_contribution_of_v_i_to_y:

    # Plots the sorted average of the contribution of each v_i to the output of y. (See text of the paper for a definition
    # of the "contribution" of v_i to y.



    v_tilde_norms_list = np.reshape(v_tilde_norms_list, [number_of_models, number_of_checkpoints, repetitions, batch_size, 228, seq_len])
    max_w_indices_list = np.reshape(max_w_indices_list, [number_of_models, number_of_checkpoints, repetitions, batch_size, 228])


    k = np.mean(np.flip(np.sort(v_tilde_norms_list, axis=5), axis=5), axis=(2, 3, 4))


    mean = np.mean(k, 0)
    sem = np.std(k, 0)/np.sqrt(k.shape[0])

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)


    plt.errorbar(
                 np.arange(seq_len) + 1,
                 mean[0, :],
                 yerr=sem[0, :],
                 capsize=5,
                 marker='o',
                 color='k',
                 label=0,
    )

    plt.errorbar(
        np.arange(seq_len) + 1,
        mean[-1, :],
        yerr=sem[-1, :],
        capsize=5,
        marker='o',
        color=prettyplot.colors['blue'],
        label=40,
    )
    prettyplot.no_box()
    prettyplot.xlabel('rank')
    plt.ylim([0, 9])
    plt.xticks([1, 10, 20])
    plt.legend(frameon=False)
    prettyplot.ylabel('contribution of $v_i$ to y')
    plt.savefig('Plots/ContributionOf_v_i_to_y.pdf')
    plt.show()

    for i in range(20):
        s = stats.ttest_rel(k[:, -1, i], k[:, 0, i])
        if s.pvalue < 0.05:
            print(f'v_{i} contribution before and after training. p =', s.pvalue)
            print('\tFull stats:' + str(s))
            print(f'\tMean before = {np.mean(k[:, -1, i])}. Mean after = {np.mean(k[:, 0, i])}')


plot_permutation_invariance_test = True
if plot_permutation_invariance_test:
    # Tests the permutation invariance of the scrambled attention layer.
    # Supposing the i is the index of the largest softmax weight and sigma is a permutation, it tests the chance that
    # if the keys are permuted by sigma, the new largest weight will be sigma(i)
    # The permutation_invariance_bit_mask has shape [number_of_models, number_of_checkpoints, repetitions, batch_size, tgt_seq_len]
    # The dimensions repetitions, batch_size and tgt_seq_len are averaged over to produce a percent permutation invariance.
    # Because the sequence length is 20, chance levels are 1/20 = 5%.

    # probability that the layer is permutation invariant. Shape = [number_of_models, number_of_checkpoints]
    p_permutation_invariant = np.mean(permutation_invariance_bit_mask, axis=(2, 3, 4))

    for i in range(number_of_checkpoints):
        s = stats.ttest_rel(p_permutation_invariant[:, i], np.ones(shape=(p_permutation_invariant[:, i].shape)) * 0.05)
        print(f'Permutation invariance for epoch {[0, 10, 20, 30, 40][i]}: p = {s.pvalue}')
        print(f'\tFull stats: {s}')
        print(f'\tmean = {np.mean(p_permutation_invariant[:, i])}')

    mean = np.mean(p_permutation_invariant, 0)
    sem = np.std(p_permutation_invariant, 0) / np.sqrt(p_permutation_invariant.shape[0])

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.errorbar(
        np.arange(len(mean)) * 10,
        mean * 100,
        yerr=sem * 100,
        capsize=5,
        marker='o',
        color='k',
    )
    plt.axhline(1/20 * 100, color='k', linestyle='--')
    prettyplot.no_box()
    prettyplot.xlabel('epoch')
    prettyplot.ylabel('% same selected weight')
    plt.xticks([0, 10, 20, 30, 40])
    plt.ylim([0, 10])
    plt.savefig('Plots/PermutationTest.pdf')
    plt.show()

