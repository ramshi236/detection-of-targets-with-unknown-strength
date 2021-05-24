import numpy as np
from numpy.linalg import multi_dot, det
from scipy.ndimage.filters import convolve
import imageio
import pickle
import spectral.io.envi as envi

imageio.plugins.freeimage.download()


def hyperMean(HSI):
    _, _, B = HSI.shape
    W = np.ones((3, 3)) * 1 / 8
    W[1, 1] = 0
    m = np.zeros_like(HSI)
    for i in range(B):
        m[:, :, i] = convolve(HSI[:, :, i], W)
    return m


def hyperCov(HSI, m):
    M, N, B = HSI.shape
    R = np.zeros((B, B))
    for j in range(M):
        for k in range(N):
            z = (HSI[j, k, :] - m[j, k, :])
            R = R + np.outer(z, z) / ((N - 1) * (M - 1))
    return R


def calculate_TPR_FPR(NT_hist, WT_hist, bins):
    # Total
    total_WT = np.sum(WT_hist)
    total_NT = np.sum(NT_hist)
    # Cumulative sum
    cum_TP = 0
    cum_FP = 0
    # TPR and FPR list initialization
    TPR_list = []
    FPR_list = []
    # Iteratre through all values of x
    for i in range(len(bins)):
        # We are only interested in non-zero values of bad
        # if WT_hist[i] > 0:
        cum_TP += WT_hist[len(bins) - 1 - i]
        cum_FP += NT_hist[len(bins) - 1 - i]
        FPR = cum_FP / total_NT
        TPR = cum_TP / total_WT
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    return TPR_list, FPR_list


def getAUC(TPR_list, FPR_list):
    thresholds = [0.1, 0.01, 0.001]
    auc = [0,0,0]
    for j, thresh in enumerate(thresholds):
        idx_thresh = np.argmax(np.asarray(FPR_list) > thresh)
        auc[j] = np.trapz(TPR_list[:idx_thresh], FPR_list[:idx_thresh])
        auc[j] = (auc[j] - 0.5 * thresh ** 2) / (thresh - 0.5 * thresh ** 2)
    return auc


def get_characteristic_ratio(HSI, R, t, m, **model_dict):
    if not (isinstance(model_dict["algorithm"], list)) and model_dict["algorithm"] in [
        "Clairvoyant replacement multivariate-t model",
        "Clairvoyant replacement gaussian model",
        "Veritas replacement multivariate-t model",
        "Veritas replacement gaussian model",
        "Veritas replacement multivariate-t model"]:
        _, _, B = HSI.shape
        mu = getMean(HSI)
        A_t = multi_dot([t - mu, R, t - mu])
        a_0 = (2 * B + A_t) ** -0.5
        ratio = model_dict["target_strength"] / a_0
        print("a_0 = " + str(a_0))
        print("a/a_0 = " + str(ratio))
        return ratio
    elif isinstance(model_dict["algorithm"], list):
        _, _, B = HSI.shape
        # A_t = multi_dot([t - m, R, t - m])
        # a_0 = (2 * B + A_t) ** -0.5
        # ratio = model_dict["target_strength"] / a_0
        # print("For replacement models")
        # print("a_0 = " + str(a_0))
        # print("a/a_0 = " + str(ratio))
        a_0 = np.linalg.multi_dot([t, R, t]) ** -0.5
        ratio = model_dict["target_strength"] / a_0
        print("For Additive models")
        print("a_0 = " + str(a_0))
        print("a/a_0 = " + str(ratio))
        return ratio
    else:
        a_0 = np.linalg.multi_dot([t, R, t]) ** -0.5
        ratio = model_dict["target_strength"] / a_0
        print("a_0 = " + str(a_0))
        print("a/a_0 = " + str(ratio))
        return ratio


def get_statistical_model(**model_dict):
    path = model_dict["data_location"] + "\\HyMap"
    data_ref = envi.open(path + "\\self_test_refl.hdr", path + "\\self_test_refl.img")
    HSI = np.array(data_ref.load())
    t = HSI[model_dict["target_location"]]
    if model_dict["recreate_statistical_model"]:
        m = hyperMean(HSI)
        R = np.linalg.inv(hyperCov(HSI, m))
        with open("statistical_model", 'wb') as f:
            pickle.dump([R, m], f)
        return HSI, R, m, t
    else:
        with open("statistical_model", 'rb') as f:
            R, m = pickle.load(f)
        return HSI, R, m, t


def is_tdist(algorithm):
    if algorithm in ["GLRT", "LMP", "EC_FTMF",
                     "Clairvoyant additive multivariate-t model",
                     "Clairvoyant replacement multivariate-t model",
                     "Veritas additive multivariate-t model",
                     "Veritas replacement multivariate-t model"]:
        return True
    else:
        return False


def getMean(HSI):
    # "getting the mean pixel vector for all the HSI"
    M, N, B = HSI.shape
    HSI = np.reshape(HSI, (M * N, B))  # each element is B size vector
    mu = np.sum(HSI, axis=0) / (M * N)
    return mu
