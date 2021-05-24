import numpy as np
from numpy.linalg import multi_dot, det
from scipy.ndimage.filters import convolve
from functions import getMean


def histogram_by_alg(HSI, R, t, m, **model_dict):
    if model_dict["algorithm"] == "Clairvoyant additive multivariate-t model":
        return clairvoyant_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "Clairvoyant replacement gaussian model":
        return clair_replacement_gaussian_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "Clairvoyant replacement multivariate-t model":
        return clair_replacement_tdist_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "Veritas additive multivariate-t model":
        return Veritas_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "Veritas replacement gaussian model":
        return Veritas_replacement_gaussian_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "Veritas replacement multivariate-t model":
        return Veritas_replacement_tdist_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "LMP":
        return LMP_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "GLRT":
        return GLRT_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "AMF":
        return AMF_alg(HSI, R, t, m, **model_dict)
    elif model_dict["algorithm"] == "ACE":
        return ACE_alg(HSI, R, t, m, **model_dict)
    if model_dict["algorithm"] == "EC_FTMF":
        return EC_FTMF_alg(HSI, R, t, m, **model_dict)
    else:  # model_dict["algorithm"] == "RX":
        return RX_alg(HSI, R, t, m, **model_dict)


def clairvoyant_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    v = model_dict["degrees_of_freedom"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    for j in range(M):
        for k in range(N):
            diff = HSI[j, k, :] - m[j, k, :]
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            NT[j, k] = (p * multi_dot([t, R, diff]) - 0.5 * (p ** 2) * multi_dot([t, R, t])) * Fv_x
            diff = diff + p * t
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            WT[j, k] = (p * multi_dot([t, R, diff]) - 0.5 * (p ** 2) * multi_dot([t, R, t])) * Fv_x
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def clair_replacement_gaussian_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = (HSI[target_loc] - p * t) * (1 - p) ** -1
    for j in range(M):
        for k in range(N):
            xm_diff = HSI[j, k, :] - m[j, k, :]
            tm_diff = t - m[j, k, :]
            A_x = multi_dot([xm_diff, R, xm_diff])
            NT[j, k] = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * p) * A_x
            xm_diff = HSI[j, k, :] * (1 - p) - m[j, k, :] + p * t
            A_x = multi_dot([xm_diff, R, xm_diff])
            WT[j, k] = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * p) * A_x
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def clair_replacement_tdist_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    v = model_dict["degrees_of_freedom"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    mu = getMean(HSI)
    A_t = multi_dot([t - mu, R, t - mu])
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = (HSI[target_loc] - p * t) * (1 - p) ** -1
    for j in range(M):
        for k in range(N):
            xm_diff = HSI[j, k, :] - m[j, k, :]
            tm_diff = t - m[j, k, :]
            A_x = multi_dot([xm_diff, R, xm_diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            D_a = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * p) * A_x
            NT[j, k] = (D_a - 0.5 * p * A_t) * Fv_x
            xm_diff = HSI[j, k, :] * (1 - p) - m[j, k, :] + p * t
            A_x = multi_dot([xm_diff, R, xm_diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            D_a = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * p) * A_x
            WT[j, k] = (D_a - 0.5 * p * A_t) * Fv_x
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def Veritas_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    v = model_dict["degrees_of_freedom"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    n = model_dict["sigmas"]  # number of sigmas
    for j in range(M):
        for k in range(N):
            diff = HSI[j, k, :] - m[j, k, :]
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            NT[j, k] = Fv_x * (multi_dot([t, R, diff]) - 0.5 * n * multi_dot([t, R, t]) ** 0.5)
            diff = diff + p * t
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            WT[j, k] = Fv_x * (multi_dot([t, R, diff]) - 0.5 * n * multi_dot([t, R, t]) ** 0.5)

    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def Veritas_replacement_gaussian_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    p = model_dict["target_strength"]
    n = model_dict["sigmas"]
    M, N, B = HSI.shape
    mu = getMean(HSI)
    A_t = multi_dot([t - mu, R, t - mu])
    a_0 = (2 * B + A_t) ** -0.5
    a = min(1, n * a_0)
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    for j in range(M):
        for k in range(N):
            xm_diff = HSI[j, k, :] - m[j, k, :]
            tm_diff = t - m[j, k, :]
            A_x = multi_dot([xm_diff, R, xm_diff])
            NT[j, k] = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * a) * A_x
            xm_diff = xm_diff + p * t
            tm_diff = tm_diff + p * t
            A_x = multi_dot([xm_diff, R, xm_diff])
            WT[j, k] = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * a) * A_x
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def Veritas_replacement_tdist_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    v = model_dict["degrees_of_freedom"]
    p = model_dict["target_strength"]
    n = model_dict["sigmas"]
    M, N, B = HSI.shape
    mu = getMean(HSI)
    A_t = multi_dot([t - mu, R, t - mu])
    a_0 = (2 * B + A_t) ** -0.5
    a = min(1, n * a_0)
    print(str(a))
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = (HSI[target_loc] - p * t) * (1 - p) ** -1
    for j in range(M):
        for k in range(N):
            xm_diff = HSI[j, k, :] - m[j, k, :]
            tm_diff = t - m[j, k, :]
            A_x = multi_dot([xm_diff, R, xm_diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            D_a = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * a) * A_x
            NT[j, k] = (D_a - 0.5 * a * A_t) * Fv_x
            xm_diff = HSI[j, k, :] * (1 - p) - m[j, k, :] + p * t
            A_x = multi_dot([xm_diff, R, xm_diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            D_a = multi_dot([tm_diff, R, xm_diff]) - (1 - 0.5 * a) * A_x
            WT[j, k] = (D_a - 0.5 * a * A_t) * Fv_x
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def LMP_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    v = model_dict["degrees_of_freedom"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    for j in range(M):
        for k in range(N):
            diff = HSI[j, k, :] - m[j, k, :]
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            NT[j, k] = multi_dot([t, R, diff]) * Fv_x
            diff = diff + p * t
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x))
            WT[j, k] = multi_dot([t, R, diff]) * Fv_x
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def GLRT_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    v = model_dict["degrees_of_freedom"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    for j in range(M):
        for k in range(N):
            diff = HSI[j, k, :] - m[j, k, :]
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x)) ** 0.5
            NT[j, k] = multi_dot([t, R, diff]) * Fv_x
            diff = diff + p * t
            A_x = multi_dot([diff, R, diff])
            Fv_x = ((v - 1) / (v - 2 + A_x)) ** 0.5
            WT[j, k] = multi_dot([t, R, diff]) * Fv_x
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def AMF_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    for j in range(M):
        for k in range(N):
            diff = HSI[j, k, :] - m[j, k, :]
            NT[j, k] = multi_dot([t, R, diff])
            diff = diff + p * t
            WT[j, k] = multi_dot([t, R, diff])
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def ACE_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    for j in range(M):
        for k in range(N):
            diff = HSI[j, k, :] - m[j, k, :]
            A_x = multi_dot([diff, R, diff])
            NT[j, k] = multi_dot([t, R, diff]) / A_x ** 1 / 2
            diff = diff + p * t
            A_x = multi_dot([diff, R, diff])
            WT[j, k] = multi_dot([t, R, diff]) / A_x ** 1 / 2
    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def RX_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    p = model_dict["target_strength"]
    M, N, B = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    for j in range(M):
        for k in range(N):
            diff = HSI[j, k, :] - m[j, k, :]
            NT[j, k] = multi_dot([diff, R, diff])
            diff = diff + p * t
            WT[j, k] = multi_dot([diff, R, diff])

    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


def EC_FTMF_alg(HSI, R, t, m, **model_dict):
    target_loc = model_dict["target_location"]
    v = model_dict["degrees_of_freedom"]
    p = model_dict["target_strength"]
    M, N, d = HSI.shape
    NT = np.zeros((M, N))
    WT = np.zeros((M, N))
    HSI[target_loc] = HSI[target_loc] - p * t
    mu = getMean(HSI)
    A = multi_dot([t - mu, R, t - mu]) + v - 2
    for j in range(M):
        for k in range(N):
            # No target test
            xm_diff = HSI[j, k, :] - m[j, k, :]
            tm_diff = t - m[j, k, :]
            xt_diff = HSI[j, k, :] - t
            # A = multi_dot([tm_diff, R, tm_diff]) + (v - 2)
            B = (1 - v / d) * multi_dot([xt_diff, R, tm_diff])
            C = (-v / d) * multi_dot([xt_diff, R, xt_diff])
            a_hat = 1 - (-B + ((B ** 2) - 4 * A * C) ** 0.5) / (2 * A)
            w = xm_diff - a_hat * tm_diff
            J = multi_dot([w, R, w]) / (v - 2)
            Q1 = 1 + J * (1 - a_hat) ** -2
            Q2 = 1 + J
            NT[j, k] = -d * np.log(1 - a_hat) - 0.5 * (d + v) * np.log(Q1 / Q2)

            xm_diff = HSI[j, k, :] * (1 - p) - m[j, k, :] + p * t
            xt_diff = HSI[j, k, :] * (1 - p) - t + p * t
            # A_x = multi_dot([xm_diff, R, xm_diff])
            # A = multi_dot([tm_diff, R, tm_diff]) + (v - 2)
            B = (1 - v / d) * multi_dot([xt_diff, R, tm_diff])
            C = (-v / d) * multi_dot([xt_diff, R, xt_diff])
            a_hat = 1 - (-B + ((B ** 2) - 4 * A * C) ** 0.5) / (2 * A)
            w = xm_diff - a_hat * tm_diff
            J = multi_dot([w, R, w]) / (v - 2)
            Q1 = 1 + J * (1 - a_hat) ** -2
            Q2 = 1 + J
            WT[j, k] = -d * np.log(1 - a_hat) - 0.5 * (d + v) * np.log(Q1 / Q2)

    NT = NT.flatten()
    WT = WT.flatten()
    min1 = min(min(NT), min(WT))
    max1 = max(max(NT), max(WT))
    bins = np.linspace(min1, max1, 1000)
    NT_hist, _ = np.histogram(NT, bins)
    WT_hist, _ = np.histogram(WT, bins)
    bins = bins[1:]
    return NT_hist, WT_hist, bins


