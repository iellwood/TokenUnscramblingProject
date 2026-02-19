import numpy as np
import loadcheckpoint

checkpoints = [0, 64410, 128820, 193230, 257640]#, 322050]
separate_models = loadcheckpoint.paths_separate_KV_scramble_matrix

nlls_vanilla_transformer = np.zeros((4, 5))
for replicate in range(4):
    for checkpoint_index in range(5):
        model_path = loadcheckpoint.get_control_model_checkpoint(replicate, checkpoints[checkpoint_index])
        model = loadcheckpoint.get_model(model_path, False, False, False)
        nll = loadcheckpoint.get_model_test_nll(model, batch_size=32, repetitions=10)
        nlls_vanilla_transformer[replicate, checkpoint_index] = nll



nlls_modified_model = np.zeros((15, 5))
for replicate in range(15):
    for checkpoint_index in range(5):
        model_path = loadcheckpoint.get_separate_KV_scramble_matrix_checkpoint(replicate, checkpoints[checkpoint_index], True)
        model = loadcheckpoint.get_model(model_path, True, True, False)
        nll = loadcheckpoint.get_model_test_nll(model, batch_size=32, repetitions=10)
        nlls_modified_model[replicate, checkpoint_index] = nll

np.savez('ModelData/nlls.npz', nlls_modified_model=nlls_modified_model, nlls_vanilla_transformer=nlls_vanilla_transformer)
