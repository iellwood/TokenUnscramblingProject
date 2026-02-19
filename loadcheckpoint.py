"""
Code for the paper

"Token unscrambling in fixed-weight biological models of transformer attention"
I.T. Ellwood 2026

This file includes simple functions for getting paths to checkpoints and computing the test NLL
"""


import torch
from Model.BatchFactory import BatchFactory
from Model.ModifiedTransformer import ModifiedTransformer as Model
assert torch.cuda.is_available(), "CUDA is not available. Unable to load model"

import numpy as np

def get_model(checkpoint_path, use_modified_layer, separate_scramble_matrix_for_K_and_V, use_test_data=False):
    device = 'cuda'

    if not use_test_data:
        batch_factory = BatchFactory(
            "iwslt_data/german_ids.npy",
            "iwslt_data/english_ids.npy",
            "Tokenizers/bytelevel_tokenizer_GERMAN.json",
            "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
            device=device
        )
    else:
        batch_factory = BatchFactory(
            "iwslt_data/german_ids_TEST.npy",
            "iwslt_data/english_ids_TEST.npy",
            "Tokenizers/bytelevel_tokenizer_GERMAN.json",
            "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
            device=device
        )

    model = Model(
        src_vocab_size=10000,
        trg_vocab_size=10000,
        src_sequence_length=batch_factory.get_src_max_sequence_length(),
        tgt_sequence_length=batch_factory.get_tgt_max_sequence_length() + 1,  # The extra one here is for autoregression
        FA_sequence_length=20,
        use_modified_layer=use_modified_layer,
        separate_scramble_matrix_for_K_and_V=separate_scramble_matrix_for_K_and_V,
        embed_dim=512,
        num_heads=8,
        number_of_layers=6,
        modified_layer=3,
        dim_feedforward=2048,
        device=device
    )

    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    return model


def get_matrices_from_model(model, normalize=False):
    layer = model.layers[2].FALayer
    M_K = layer.M_K
    M_V = layer.M_V

    if normalize:
        M_K = layer.normalize_scramble_matrix(M_K)
        M_V = layer.normalize_scramble_matrix(M_V)

    M_K = M_K.detach().cpu().numpy()
    M_V = M_V.detach().cpu().numpy()

    return M_K, M_V


def get_control_model_checkpoint(model_number, step_number):
    return 'ModelCheckpoints/FAModel_Control_replicate_' + str(model_number) + '/Checkpoint_at_step_' + str(step_number) + '.pth'

def get_separate_KV_scramble_matrix_checkpoint(model_number, step_number, separate_M_k_and_M_v):
    if separate_M_k_and_M_v:
        return 'ModelCheckpoints/FAModel_SeparateScrambleMatrices_replicate_' + str(model_number) + '/Checkpoint_at_step_' + str(step_number) + '.pth'
    else:
        return 'ModelCheckpoints/FAModel_SingleScrambleMatrix_replicate_' + str(model_number) + '/Checkpoint_at_step_' + str(step_number) + '.pth'


def get_model_test_nll(model, batch_size=32, repetitions=100):
    NLLs = []
    batch_factory = BatchFactory(
        "iwslt_data/german_ids_TEST.npy",
        "iwslt_data/english_ids_TEST.npy",
        "Tokenizers/bytelevel_tokenizer_GERMAN.json",
        "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
        device='cuda',
    )


    model.train(False)
    for repetition in range(repetitions):
        src, tgt, tgt_shifted = batch_factory.get_batch(batch_size)
        NLL = model.compute_NLL(src, tgt, tgt_shifted)
        NLLs.append(NLL.detach().cpu().numpy() * 1)
    return np.mean(NLLs)

def get_model_train_nll(model, batch_size=32, repetitions=100):
    NLLs = []
    batch_factory = BatchFactory(
        "iwslt_data/german_ids.npy",
        "iwslt_data/english_ids.npy",
        "Tokenizers/bytelevel_tokenizer_GERMAN.json",
        "Tokenizers/bytelevel_tokenizer_ENGLISH.json",
        device='cuda',
    )


    model.train(True)
    for repetition in range(repetitions):
        src, tgt, tgt_shifted = batch_factory.get_batch(batch_size)
        NLL = model.compute_NLL(src, tgt, tgt_shifted)
        NLLs.append(NLL.detach().cpu().numpy() * 1)
    return np.mean(NLLs)
