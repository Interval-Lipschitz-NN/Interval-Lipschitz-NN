from CLEVER_python.CLEVER import clever_u
import numpy as np

def CLEVER_Lipschitz_Default_Dataset(X_test, myPred, classifier, Nb, Ns, linf_radius, pool_size):
    clever_x_test = np.array([X_test[0].data.numpy()])
    clever_y_test = myPred[0]

    # Get CLEVER score & max gradient
    clever_score, loc_list = clever_u(classifier, clever_x_test[-1], Nb, Ns, linf_radius, norm=np.inf, pool_factor=pool_size)
    lipschitz = max(loc_list)
    print("CLEVER score is " + str(clever_score))
    print("Lipschitz constant in CLEVER is " + str(lipschitz) + "\n")

    return lipschitz, clever_x_test, clever_y_test

def CLEVER_Lipschitz_Special(X_test, myPred, classifier, Nb, Ns, linf_radius, pool_size):
    clever_x_test = X_test
    clever_y_test = myPred[0]

    # Get CLEVER score & max gradient
    clever_score, loc_list = clever_u(classifier, clever_x_test, Nb, Ns, linf_radius, norm=np.inf, pool_factor=pool_size)
    lipschitz = max(loc_list)
    print("CLEVER score is " + str(clever_score))
    print("Lipschitz constant in CLEVER is " + str(lipschitz) + "\n")

    x_return = np.swapaxes(clever_x_test.reshape(clever_x_test.shape[0],-1), 0, 1)
    return lipschitz, x_return, clever_y_test