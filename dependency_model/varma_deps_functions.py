from dependency_model.learn_deps import learn_structure, get_deps_from_inverse_sig
import numpy as np

def get_varma_edges(L_dev, thresh=1.5):
    J_hat = learn_structure(L_dev)
    deps_hat = get_deps_from_inverse_sig(J_hat, thresh=thresh)
    #remove repeated edges
    varma_deps = []
    for i,j in deps_hat:
        if i < j:
            varma_deps.append((i,j))
    return varma_deps

def get_varma_with_gold_edges(L_dev, Y_dev, thresh=1.5):
    Complete_dev = np.concatenate((L_dev, np.array([Y_dev]).T), axis=1)
    J_hat = learn_structure(Complete_dev)
    deps_hat = get_deps_from_inverse_sig(J_hat, thresh=thresh)
    #remove repeated edges
    varma_deps = []
    Y_dev_pos = L_dev.shape[1]
    for i,j in deps_hat:
        if i!=Y_dev_pos and j!=Y_dev_pos:
            if i < j:
                varma_deps.append((i,j))
    return varma_deps