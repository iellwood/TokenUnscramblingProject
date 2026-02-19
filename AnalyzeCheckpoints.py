"""
Code for the paper

"Token unscrambling in fixed-weight biological models of transformer attention"
I.T. Ellwood 2026

This file contains most of the analysis performed on the checkpoints of the trained models. The results
of the analysis are saved in

    ModelData/CheckpointAnalysisResults.obj

and are used by

    PlotScripts/PlotDataFiguresForPaper.py



"""

import torch
import loadcheckpoint
import numpy as np
from Model.BatchFactory import BatchFactory
import time
import pickle
import warnings
warnings.filterwarnings(
    "ignore",
    message="1Torch was not compiled with flash attention.",
    category=UserWarning
)

device = 'cuda'
batch_size = 32
#checkpoints = [0, 64410, 128820, 193230, 257640, 322050]
#number_of_models = 15
#number_of_checkpoints = 5
PCA_dimensions = np.arange(64) + 1
seq_len = 20
vec_dim = 64
repetitions = 10

def reshape_list(x, pca=True):
    x = np.array(x)
    if pca:
        s = x.shape
        new_shape = [number_of_models, number_of_checkpoints, len(PCA_dimensions)] + s[1:]
        return np.reshape(x, new_shape)
    else:
        s = x.shape
        new_shape = [number_of_models, number_of_checkpoints] + s[1:]
        return np.reshape(x, new_shape)


def get_data_for_PCA(path, repetitions=10, center=False):
    """
    This function computes the data required for a PCA reduction of the scramble matrices.
    The main output of the function is a similarity matrix S such that S.T @ C @ S is diagonal,
    where C is the covariance matrix for the query or the value/key of a single token.
    In the process of computing this matrix, it also computes the explained variance for each PCA component and outputs
    the scramble matrix for each model.
    :param path: Path to the model checkpoint for analysis
    :param repetitions: Number of batches of queries/keys/values to collect. The total number of samples for computing the covariance matrix is batch_size * reprititions.
    :param center: Whether to center the data before PCA analysis. In standard PCA, the data is always centered, This does not seem natural for this problem, however.
    :return: M_K, M_V, eigenvectors_q, eigenvectors_k, eigenvectors_v, explained_variance_query, explained_variance_keys, explained_variance_values
    """
    model = loadcheckpoint.get_model(path, True, separate_scramble_matrix_for_K_and_V=True)
    model.train(False)

    M_K = model.layers[2].FALayer.M_K.detach().cpu().numpy()
    M_V = model.layers[2].FALayer.M_V.detach().cpu().numpy()

    batch_factory = BatchFactory(
        "iwslt_data/german_ids.npy",
        "iwslt_data/english_ids.npy",
        "Tokenizers/bytelevel_tokenizer_GERMAN.json",
        "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
        device=device
    )

    t_0 = time.time()

    q_inputs = []
    k_inputs = []
    v_inputs = []

    for rep in range(repetitions):
        src, tgt, tgt_shifted = batch_factory.get_batch(batch_size)
        NLL = model.compute_NLL(src, tgt, tgt_shifted)
        qkv_input = model.layers[2].FALayer.last_qkv_input
        q_input = qkv_input[0].clone().detach().cpu().numpy()
        k_input = qkv_input[1].clone().detach().cpu().numpy()
        v_input = qkv_input[2].clone().detach().cpu().numpy()

        q_inputs.append(q_input.transpose([1, 0, 2])[:, :20, :]) # The :20 here is to make the size of the q batch equal the k/v batch by onl using the first 20 tokens in the decoder.
        k_inputs.append(k_input.transpose([1, 0, 2]))
        v_inputs.append(v_input.transpose([1, 0, 2]))

    # All of the x_inputs have shape = [repetitions, batch_size, seq_len, vec_dim]
    q_inputs = np.array(q_inputs)
    k_inputs = np.array(k_inputs)
    v_inputs = np.array(v_inputs)

    # Reshape the vectors by combining the batch dimension and repetitions into one large batch
    # New shape = [batch_size * repetitions, seq_len, vec_dim]
    q_inputs = np.reshape(q_inputs, [-1, q_inputs.shape[2], q_inputs.shape[3]])
    k_inputs = np.reshape(k_inputs, [-1, k_inputs.shape[2], k_inputs.shape[3]])
    v_inputs = np.reshape(v_inputs, [-1, v_inputs.shape[2], v_inputs.shape[3]])

    eigenvectors_k = []
    eigenvectors_v = []

    explained_variance_keys = np.zeros(shape=[k_inputs.shape[1], 64])
    explained_variance_values = np.zeros(shape=[k_inputs.shape[1], 64])

    for token_index in range(k_inputs.shape[1]):


        # x is a collection of keys from a single token. It has shape [batch_size * repetitions, vec_dim]
        # the covariance matrix has shape [vec_dim, vec_dim]
        x = k_inputs[:, token_index, :]
        if center: x = x - np.mean(k_inputs[:, token_index, :], 0, keepdims=True)
        covariance_matrix = np.matmul(x.T, x)

        # This section computes the Similarity matrix S which transforms the values in the basis of eigenvectors of the
        # covariance matrix. (S.T @ covariance_matrix @ S is diagonal.)
        #
        # Note that S = eigenvectors, the matrix whose rows are the normalized, orthogonalized
        # eigenvectors. S is also sorted so that the first eigenvector S[:, 0] is the one with the largest eigenvalue
        # of the covariance matrix (i.e. the one with the largest explained variance).
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        I = np.argsort(eigenvalues) # argsort the eigenvalues
        I = np.flip(I) # We want that the first eigenvalue is the *largest*
        eigenvalues = eigenvalues[I]
        eigenvalues = eigenvalues/np.sum(eigenvalues)
        explained_variance_keys[token_index, :] = eigenvalues
        eigenvectors = eigenvectors[:, I]  # Sort the eigenvectors by the same sort as the eigenvalues.
        eigenvectors_k.append(eigenvectors) # Save the eigenvectors in the list of all key eigenvector matrices

        # This repeats the analysis above for the values
        x = v_inputs[:, token_index, :]
        if center: x = x - np.mean(v_inputs[:, token_index, :], 0, keepdims=True)
        covariance_matrix = np.matmul(x.T, x)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        I = np.argsort(eigenvalues)
        I = np.flip(I)
        eigenvalues = eigenvalues[I]
        eigenvalues = eigenvalues/np.sum(eigenvalues)
        explained_variance_values[token_index, :] = eigenvalues
        eigenvectors = eigenvectors[:, I]
        eigenvectors_v.append(eigenvectors)

    # Make an array of seq_len eigenvector matrices. Overall shape = [seq_len, vec_dim, vec_dim]
    eigenvectors_k = np.array(eigenvectors_k)
    eigenvectors_v = np.array(eigenvectors_v)

    # Finally we perform the analysis for the keys. Note that there is only one key, so we do not loop over the token index.
    x = np.reshape(q_inputs, [-1, vec_dim])
    if center: x - np.mean(q_inputs, 0, keepdims=True)
    covariance_matrix = np.matmul(x.T, x)
    eigenvalues, eigenvectors_q = np.linalg.eig(covariance_matrix)
    I = np.argsort(eigenvalues)
    I = np.flip(I)
    eigenvalues = eigenvalues[I]
    explained_variance_query = eigenvalues/np.sum(eigenvalues)
    eigenvectors_q = np.array(eigenvectors_q)

    return M_K, M_V, eigenvectors_q, eigenvectors_k, eigenvectors_v, explained_variance_query, explained_variance_keys, explained_variance_values

def perform_PCA_analysis(M_K, M_V, eigenvectors_q, eigenvectors_k, eigenvectors_v, PCA_dimension):
    """
    Performs a PCA reduction of the scramble matrices using the results of the function get_data_for_PCA
    :param M_K: The original key scramble matrix
    :param M_V: The original value scramble matrix
    :param eigenvectors_q: The similarity transformation to the basis of eigenvalues of the covariance matrix for the queries.
    :param eigenvectors_k: An array of seq_len similarity transformations to the basis of eigenvalues of the covariance matrix of the keys
    :param eigenvectors_v: An array of seq_len similarity transformations to the basis of eigenvalues of the covariance matrix of the values
    :param PCA_dimension: Number of dimensions to keep after the PCA reduction
    :return: M_K_projected, M_K_averaged_over_vec_dim, M_V_projected, M_V_averaged_over_vec_dim
    """

    # Before applying PCA, we find the permutation, I, that will sort the rows by the position of their largest value
    M_K_no_PCA_averaged_over_vec_dim = M_K.reshape([seq_len, vec_dim, seq_len, vec_dim])
    M_K_no_PCA_averaged_over_vec_dim = np.mean(np.abs(M_K_no_PCA_averaged_over_vec_dim), axis=(1, 3))

    # Find the location of the largest value in each row
    # Note that axis 0 is the rows as we have not transposed the scramble matrix so that it acts on the left of vectors
    index_of_largest_value = np.argmax(M_K_no_PCA_averaged_over_vec_dim, 0)
    permutation_of_row_indices = np.argsort(index_of_largest_value)
    
    


    k_projection = np.zeros(shape=(seq_len, seq_len, vec_dim, PCA_dimension))
    v_projection = np.zeros(shape=(seq_len, seq_len, vec_dim, PCA_dimension))
    q_projection = np.zeros(shape=(seq_len, seq_len, vec_dim, PCA_dimension))

    for token_index in range(seq_len):
        k_projection[token_index, token_index, :, :] = eigenvectors_k[token_index, :, :PCA_dimension]
        v_projection[token_index, token_index, :, :] = eigenvectors_v[token_index, :, :PCA_dimension]
        q_projection[token_index, token_index, :, :] = eigenvectors_q[:, :PCA_dimension]

    # This makes the shape [seq_len, vec_dim, seq_len, PCA_Dimension]
    q_projection = np.transpose(q_projection, [0, 2, 1, 3])
    k_projection = np.transpose(k_projection, [0, 2, 1, 3])
    v_projection = np.transpose(v_projection, [0, 2, 1, 3])

    # Makes the shape [seq_len * vec_dim, seq_len * PCA_Dimension].
    # Compare this with the shape of the scramble matrices [seq_len * vec_dim, seq_len * vec_dim]
    q_projection = np.reshape(q_projection, [q_projection.shape[0] * q_projection.shape[1], q_projection.shape[2] * q_projection.shape[3]])
    k_projection = np.reshape(k_projection, [k_projection.shape[0] * k_projection.shape[1], k_projection.shape[2] * k_projection.shape[3]])
    v_projection = np.reshape(v_projection, [v_projection.shape[0] * v_projection.shape[1], v_projection.shape[2] * v_projection.shape[3]])

    P_q = q_projection @ q_projection.T
    P_k = k_projection @ k_projection.T
    P_v = v_projection @ v_projection.T

    # This code computes the projected scramble matrices.
    # 1) Note that the scramble matrices here obey the usual machine learning convention that they act on the right v -> vM
    #    After reducing, we transpose the answer so that the matrices act on the left v -> Mv.
    # 2) Unlike with the query and keys, we only project on one side of the value scramble matrix.
    M_K_projected = (P_k @ M_K @ P_q).T  # shape = [seq_len * PCA_dimension, seq_len * PCA_Dimension]
    M_V_projected = (P_v @ M_V).T  # shape = [seq_len * vec_dim, seq_len * PCA_Dimension]

    # Converts the scramble matrices to 4-index notation
    M_K_projected_4index = M_K_projected.reshape([seq_len, vec_dim, seq_len, vec_dim])
    M_V_projected_4index = M_V_projected.reshape([seq_len, vec_dim, seq_len, vec_dim])

    # Compute the norm of each block
    M_K_averaged_over_vec_dim = np.mean(np.abs(M_K_projected_4index), axis=(1, 3))
    M_V_averaged_over_vec_dim = np.mean(np.abs(M_V_projected_4index), axis=(1, 3))

    # Permute the rows using the permutation defined above
    M_K_averaged_over_vec_dim = M_K_averaged_over_vec_dim[permutation_of_row_indices, :]
    M_V_averaged_over_vec_dim = M_V_averaged_over_vec_dim[permutation_of_row_indices, :]

    M_K_projected_4index = M_K_projected_4index[permutation_of_row_indices, :, :, :]
    M_V_projected_4index = M_V_projected_4index[permutation_of_row_indices, :, :, :]
    M_K_projected = np.reshape(M_K_projected_4index, M_K_projected.shape)
    M_V_projected = np.reshape(M_V_projected_4index, M_V_projected.shape)

    return M_K_projected, M_K_averaged_over_vec_dim, M_V_projected, M_V_averaged_over_vec_dim

def compute_weights(q, k):
    """
    Reimplements torch's softmax attention layer, but outputs the weights instead of the weights times the values.
    :param q: queries. q.shape =[batch_size, 1, query_sequence_length, vec_dim]
    :param k: keys. k.shape = [batch_size, 1, key_sequence_length, vec_dim]
    :return: weights. w.shape = [batch_size, query_sequence_length, key_sequence_length]
    """
    Q = q[:, 0, :, :]
    K = torch.permute(k[:, 0, :, :], [0, 2, 1])
    QK = torch.bmm(Q, K)/torch.sqrt(torch.tensor(Q.shape[2], device=device))
    return torch.softmax(QK, 2)

def sparsity_metric(matrix, axis):
    n = matrix.shape[axis]
    L_1 = np.sum(np.abs(matrix), axis)
    L_2 = np.sqrt(np.sum(np.square(matrix), axis)) + 1e-10
    return np.mean((L_1/L_2 - 1)/(np.sqrt(n) - 1))

def get_sorted_w_average(path, repetitions=10):
    """
    Computes the weight vector of softmax attention in the scrambled attention layer. The weights are sorted
    and then averaged over the queries. The batch_dimension is not averaged over.
    :param path: Path to a model checkpoint.
    :param repetitions: Number of batches to run
    :return: Sorted average weights w.  w.shape = [repetitions * batch_size, key_sequence_length]
    """
    model = loadcheckpoint.get_model(path, True, separate_scramble_matrix_for_K_and_V=True)
    model.train(False)

    batch_factory = BatchFactory(
        "iwslt_data/german_ids.npy",
        "iwslt_data/english_ids.npy",
        "Tokenizers/bytelevel_tokenizer_GERMAN.json",
        "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
        device=device
    )

    t_0 = time.time()

    w_list = []

    for rep in range(repetitions):
        src, tgt, tgt_shifted = batch_factory.get_batch(batch_size)
        NLL = model.compute_NLL(src, tgt, tgt_shifted)
        w = compute_weights(model.layers[2].FALayer.last_input_to_softmax_attention[0], model.layers[2].FALayer.last_input_to_softmax_attention[1])

        w = w.detach().cpu().clone().numpy()

        w = np.mean(np.sort(w, axis=-1), axis=1)  # Average over the queries, but not the batch_dimension
        w_list.append(w)

    ws = np.concatenate(w_list, axis=0)

    return ws


def permute_rows(x):
    x = torch.permute(x, [2, 1, 0])
    m, batch_size, n = x.shape
    I = torch.stack([torch.randperm(n) for _ in range(batch_size)])  # Unique permutation for each minibatch
    I = torch.stack([I for _ in range(m)])  # Identical permutations within vector dimension
    I = I.to('cuda')
    x_permuted = torch.gather(x, dim=2, index=I)
    x_permuted = torch.permute(x_permuted, [2, 1, 0])
    return x_permuted, I[0, :, :]

def test_permutation_invariance(path, repetitions=10):
    """
    Computes the weights w_i and w_i^p, the weights after a permutation of k. Letting i_max = argmax(w_i)
    and i_max^p = argmax(w_i^p), computes the probability that p(i_max) = i_max^p
    """
    model = loadcheckpoint.get_model(path, True, separate_scramble_matrix_for_K_and_V=True)
    model.train(False)

    batch_factory = BatchFactory(
        "iwslt_data/german_ids.npy",
        "iwslt_data/english_ids.npy",
        "Tokenizers/bytelevel_tokenizer_GERMAN.json",
        "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
        device=device
    )

    t_0 = time.time()

    scramble_layer = model.layers[2].FALayer

    masks = []

    for rep in range(repetitions):
        src, tgt, tgt_shifted = batch_factory.get_batch(batch_size)
        NLL = model.compute_NLL(src, tgt, tgt_shifted)
        qkv = scramble_layer.last_qkv_input

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        scramble_layer(q, k, v)
        w = compute_weights(scramble_layer.last_input_to_softmax_attention[0], scramble_layer.last_input_to_softmax_attention[1])

        k_permuted, I = permute_rows(k)
        scramble_layer(q, k_permuted, v)
        w_permuted = compute_weights(scramble_layer.last_input_to_softmax_attention[0], scramble_layer.last_input_to_softmax_attention[1])

        I = torch.stack([I for _ in range(228)], dim=1)  # Identical permutations within vector dimension

        permuted_w_permuted = torch.gather(w_permuted, index=I, dim=2)

        mask = torch.argmax(w, dim=2) == torch.argmax(permuted_w_permuted, dim=2)
        mask = mask.to(torch.float32)
        mask = mask.detach().clone().cpu().numpy()

        masks.append(mask)

    return np.array(masks)

def get_sorted_V_contribution(path, repetitions=10):
    """
    Computes the weight vector of softmax attention in the scrambled attention layer. The weights are sorted
    and then averaged over the queries. The batch_dimension is not averaged over.
    :param path: Path to a model checkpoint.
    :param repetitions: Number of batches to run
    :return: Sorted average weights w.  w.shape = [repetitions * batch_size, key_sequence_length]
    """
    model = loadcheckpoint.get_model(path, True, separate_scramble_matrix_for_K_and_V=True)
    model.train(False)

    batch_factory = BatchFactory(
        "iwslt_data/german_ids.npy",
        "iwslt_data/english_ids.npy",
        "Tokenizers/bytelevel_tokenizer_GERMAN.json",
        "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
        device=device
    )

    t_0 = time.time()

    z_list = []

    max_w_index_list = []

    for rep in range(repetitions):
        src, tgt, tgt_shifted = batch_factory.get_batch(batch_size)
        NLL = model.compute_NLL(src, tgt, tgt_shifted)
        w = compute_weights(model.layers[2].FALayer.last_input_to_softmax_attention[0], model.layers[2].FALayer.last_input_to_softmax_attention[1])

        w = w.detach().cpu().numpy()

        max_w_indices = np.argmax(w, axis=2)

        v = model.layers[2].FALayer.last_qkv_input[2].detach().cpu().numpy()

        M_V = model.layers[2].FALayer.M_V.detach().cpu().numpy()

        v = np.transpose(v, [1, 0, 2])  # [batch_size, seq_len, vec_dim]
        #v = np.reshape([v, [-1, vec_dim]])
        M_V = np.reshape(M_V, [seq_len, vec_dim, seq_len, vec_dim])

        M_V = np.transpose(M_V, [1, 3, 0, 2])  # [vec_dim, vec_dim, seq_len, seq_len]
        v_tilde = np.tensordot(v, M_V, axes=1)

        v_norm = np.linalg.norm(v_tilde, axis=2)
        v_norm = np.diagonal(v_norm, axis1=1, axis2=2)  # [batch_size, v_sequence_length, weight_index]

        zs = []
        for batch_index in range(batch_size):

            z_q_list = []
            for q_sequence_index in range(max_w_indices.shape[1]):
                z = v_norm[batch_index, :, max_w_indices[batch_index, q_sequence_index]]
                z_q_list.append(z)

            zs.append(z_q_list)

        zs = np.array(zs)

        z_list.append(zs)
        max_w_index_list.append(max_w_indices)

    v_tilde_norms = np.array(z_list)
    max_w_indices = np.array(max_w_index_list)

    return v_tilde_norms, max_w_indices



def collect_all_scramble_matrices_with_PCA(model_path_function, model_IDs, PCA_dimensions, checkpoints, repetitions):
    """
    Performs the PCA reduction analysis for all models, checkpoints and PCA dimensions.
    :param model_path_function: A function that outputs the path to a model given a model number and checkpoint. I.e. model_path = model_path_function(model_number, checkpoint)
    :param model_IDs: A list of all model ID numbers to process. This will typically be np.arange(number_of_models)
    :param PCA_dimensions: A list of all PCA dimensions to reduce to. Typically np.arange(vec_dim) + 1
    :param checkpoints: Checkpoints to process. Should be np.arange(number_of_checkpoints)
    :param repetitions: Determines the number of samples to be collected for the PCA covariance matrices (batch_size * repetitions)
    :return: M_K_proj, M_K_proj_token_average, M_V_proj, M_V_proj_token_average, explained_variance_q, explained_variance_k, explained_variance_v

    """
    number_of_models = len(model_IDs)
    number_of_checkpoints = len(checkpoints)
    number_of_PCA_dimension = len(PCA_dimensions)

    print('Computing PCA reduction')
    print('Number of models =', number_of_models)
    print('Number of checkpoints =', number_of_checkpoints)

    M_K_proj = []
    M_K_proj_token_average = []

    M_V_proj = []
    M_V_proj_token_average = []


    # Will be used to store the explained variance vs. PCA dimension.
    # Note that for the keys and values we are storing the explained variance separately for each token.
    explained_variance_q = np.zeros((number_of_models, number_of_checkpoints, vec_dim))
    explained_variance_k = np.zeros((number_of_models, number_of_checkpoints, seq_len, vec_dim))
    explained_variance_v = np.zeros((number_of_models, number_of_checkpoints, seq_len, vec_dim))

    M_K_list = []
    M_V_list = []
    eigenvectors_q = []
    eigenvectors_k = []
    eigenvectors_v = []
    for i in range(len(model_IDs)):
        M_K_list.append([])
        M_V_list.append([])
        eigenvectors_q.append([])
        eigenvectors_k.append([])
        eigenvectors_v.append([])

    print("Collecting similarity transformations for each model and checkpoint")
    for model_number in model_IDs:
        print("\tModel #", model_number)
        for checkpoint_number, checkpoint in enumerate(checkpoints):
            print('\t\tCheckpoint #', checkpoint_number)
            model_path = model_path_function(model_number, checkpoint)
            M_K, M_V, eig_q, eig_k, eig_v, ev_q, ev_k, ev_v = get_data_for_PCA(model_path, repetitions=repetitions, center=False)
            M_K_list[model_number].append(M_K)
            M_V_list[model_number].append(M_V)
            eigenvectors_q[model_number].append(eig_q)
            eigenvectors_k[model_number].append(eig_k)
            eigenvectors_v[model_number].append(eig_v)
            explained_variance_q[model_number, checkpoint_number, :] = ev_q
            explained_variance_k[model_number, checkpoint_number, :, :] = ev_k
            explained_variance_v[model_number, checkpoint_number, :, :] = ev_v

    # Initialize the array to hold all of the scramble matrices with different amounts of PCA reduction
    #M_K = np.zeros((number_of_PCA_dimension, number_of_models, number_of_checkpoints, seq_len * vec_dim, seq_len * vec_dim))
    M_K_avg = np.zeros((number_of_PCA_dimension, number_of_models, number_of_checkpoints, seq_len, seq_len))
    #M_V = np.zeros((number_of_PCA_dimension, number_of_models, number_of_checkpoints, seq_len * vec_dim, seq_len * vec_dim))
    M_V_avg = np.zeros((number_of_PCA_dimension, number_of_models, number_of_checkpoints, seq_len, seq_len))

    print('Using similarity matrices to compute PCA reduction')
    for pca_dimension_index, PCA_dimension in enumerate(PCA_dimensions):
        print(f'\tReducing to {PCA_dimension}.dimensions')

        for model_number in model_IDs:
            for checkpoint_number, checkpoint in enumerate(checkpoints):
                m_k, m_k_av, m_v, m_v_av = perform_PCA_analysis(
                    M_K_list[model_number][checkpoint_number],
                    M_V_list[model_number][checkpoint_number],
                    eigenvectors_q[model_number][checkpoint_number],
                    eigenvectors_k[model_number][checkpoint_number],
                    eigenvectors_v[model_number][checkpoint_number],
                    PCA_dimension=PCA_dimension)

                #M_K[pca_dimension_index, model_number, checkpoint_number, :, :] = m_k
                M_K_avg[pca_dimension_index, model_number, checkpoint_number, :, :] = m_k_av
                #M_V[pca_dimension_index, model_number, checkpoint_number, :, :] = m_v
                M_V_avg[pca_dimension_index, model_number, checkpoint_number, :, :] = m_v_av

    return M_K_avg, M_V_avg, explained_variance_q, explained_variance_k, explained_variance_v

def get_max_weight_and_v_tilde_norms(model_path_function, model_IDs, checkpoints, repetitions):
    print('Computing v_tilde norms and max_w_indices')
    v_tilde_norms_list = []
    max_w_indices_list = []
    for model_number in model_IDs:
        print(f'\t Model number: {model_number}.')

        for checkpoint_number, checkpoint in enumerate(checkpoints):
            model_path = model_path_function(model_number, checkpoint)
            v_tilde_norms, max_w_indices = get_sorted_V_contribution(path=model_path, repetitions=repetitions)
            v_tilde_norms_list.append(v_tilde_norms)
            max_w_indices_list.append(max_w_indices)

    v_tilde_norms_list = np.array(v_tilde_norms_list)
    max_w_indices_list = np.array(max_w_indices_list)

    return max_w_indices_list, v_tilde_norms_list

def collect_permutation_invariance_data(model_path_function, model_IDs, checkpoints, repetitions):
    print(f'Testing Permutation Invariance')
    permutation_invariance_bit_mask = []
    for model_number in model_IDs:
        print(f'\t Model number: {model_number}.')
        masks_per_checkpoint = []
        for checkpoint_number, checkpoint in enumerate(checkpoints):
            model_path = model_path_function(model_number, checkpoint)
            mask = test_permutation_invariance(path=model_path, repetitions=repetitions)
            masks_per_checkpoint.append(mask)
        permutation_invariance_bit_mask.append(masks_per_checkpoint)

    return permutation_invariance_bit_mask

def collect_softmax_weights(model_path_function, model_IDs, checkpoints, repetitions):
    print(f'Computing softmax weights')

    weights_list = []
    for model_number in model_IDs:
        print(f'\t Model number: {model_number}.')
        for checkpoint_number, checkpoint in enumerate(checkpoints):
            model_path = model_path_function(model_number, checkpoint)
            weights = get_sorted_w_average(path=model_path, repetitions=repetitions)
            weights_list.append(weights)

    softmax_w = np.array(weights_list)

    return softmax_w


def collect_and_save_model_data(save_file_name, model_path_function, model_IDs, PCA_dimensions, checkpoints, repetitions):

    # Compute the PCA reductions of the scramble matrices
    M_K_proj_token_average, M_V_proj_token_average, ev_q, ev_k, ev_v = collect_all_scramble_matrices_with_PCA(model_path_function, model_IDs, PCA_dimensions, checkpoints, repetitions)

    # Compute the amount that each value contributes to the output of the scrambled attention layer
    max_w_indices_list, v_tilde_norms_list = get_max_weight_and_v_tilde_norms(model_path_function, model_IDs, checkpoints, repetitions)

    # Test if the scrambled attention layer is invariant under permutations
    permutation_invariance_bit_mask = collect_permutation_invariance_data(model_path_function, model_IDs, checkpoints, repetitions)

    # Compute the softmax weights during training
    softmax_w = collect_softmax_weights(model_path_function, model_IDs, checkpoints, repetitions)

    save_dictionary = {
        'M_K_proj_token_average': M_K_proj_token_average,
        'M_V_proj_token_average': M_V_proj_token_average,
        'softmax_w': softmax_w,
        'checkpoints': checkpoints,
        'number_of_models': len(model_IDs),
        'number_of_checkpoints': len(checkpoints),
        'PCA_dimensions': PCA_dimensions,
        'seq_len': 20,
        'vec_dim': 64,
        'repetitions': repetitions,
        'batch_size': 32,
        'v_tilde_norms_list': v_tilde_norms_list,
        'max_w_indices_list': max_w_indices_list,
        'permutation_invariance_bit_mask': permutation_invariance_bit_mask,
        'explained variance by PCA dimension, query': ev_q,
        'explained variance by PCA dimension, key': ev_k,
        'explained variance by PCA dimension, value': ev_v,
    }

    with open('ModelData/' + save_file_name, 'wb') as file:
        pickle.dump(save_dictionary, file)


model_path_function_basic = lambda mod_num, cp: loadcheckpoint.get_separate_KV_scramble_matrix_checkpoint(mod_num, cp, separate_M_k_and_M_v=True)

collect_and_save_model_data(
    save_file_name='CheckpointAnalysisResults.obj',
    model_path_function=model_path_function_basic,
    model_IDs=np.arange(15),
    PCA_dimensions=PCA_dimensions,
    checkpoints=[0, 64410, 128820, 193230, 257640],
    repetitions=20
)

