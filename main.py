import numpy as np
import pickle
from scipy.ndimage.filters import convolve
# from dbfread import DBF
import os
import imageio
from functions import *
from algorithms import *
from sklearn import metrics

imageio.plugins.freeimage.download()
from spectral import imshow, view_cube
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

plt.rcParams.update({'font.size': 18})


def simulate(**model_dict):
    HSI, R, m, t = get_statistical_model(**model_dict)
    ratio = get_characteristic_ratio(HSI, R, t, m, **model_dict)

    if model_dict["simulation_type"] == "unknown_target_strength":
        pass
        # algorithms = model_dict["algorithm"]
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle("unknown_target_strength - AUC compression")
        # p = model_dict["target_strength"]
        # n = [(j + 1) * 0.15 for j in range(10)]
        # a_vec = np.array(n) * (p / a_0)
        #
        # # calculate "Clairvoyant" algorithm first to compere
        # NT_hist, WT_hist, bins = histogram_by_alg(HSI, R, t, m, **model_dict)
        # TPR_list, FPR_list = calculate_TPR_FPR(NT_hist, WT_hist, bins)
        # clairvoyant_AUC = 1 - getAUC(TPR_list, FPR_list)
        # clairvoyant_AUC = np.ones(len(n)) * clairvoyant_AUC
        # line, = ax2.plot(n, clairvoyant_AUC)
        # line.set_label("Clairvoyant")
        #
        # for algorithm in algorithms:  # assuming that this is a list
        #     if algorithm == "Clairvoyant":
        #         continue  # we already calculated
        #     AUC = []
        #     for a in a_vec:
        #         model_dict["algorithm"] = algorithm
        #         NT_hist, WT_hist, bins = histogram_by_alg(HSI, R, t, m, **model_dict)
        #         TPR_list, FPR_list = calculate_TPR_FPR(NT_hist, WT_hist, bins)
        #         AUC.append(getAUC(TPR_list, FPR_list))
        #     AUC = 1 - np.asarray(AUC)
        #     line, = ax1.plot(n, AUC)
        #     line.set_label(algorithm)
        #     # compared to Clairvoyant algorithm
        #     AUC = clairvoyant_AUC / AUC
        #     line, = ax2.plot(n, AUC)
        #     line.set_label(algorithm)
        # ax1.set_title("Unknown target strength", fontsize=14)
        # ax1.set_ylabel('1 - AUC', fontsize=12)
        # ax1.set_xlabel('target strength a/a_0', fontsize=12)
        # ax1.set_yscale('log', nonposy='clip')
        # ax1.grid()
        # ax1.legend()
        #
        # ax2.set_title("Unknown target strength", fontsize=14)
        # ax2.set_ylabel('1 - AUC : reltive to Clairvoyant', fontsize=12)
        # ax2.set_xlabel('target strength a/a_0', fontsize=12)
        # ax2.set_yscale('log', nonposy='clip')
        # ax2.grid()
        # ax2.legend()
        #
        # plt.show()

    elif model_dict["simulation_type"] == "plot_all_roc":  # assuming known signal strength,
        algorithms = model_dict["algorithm"]
        fig, ax = plt.subplots()
        for algorithm in algorithms:
            model_dict["algorithm"] = algorithm
            if is_tdist(model_dict["algorithm"]) and isinstance(model_dict["degrees_of_freedom"], list):
                dofs = model_dict["degrees_of_freedom"]
                for dof in dofs:
                    model_dict["degrees_of_freedom"] = dof
                    NT_hist, WT_hist, bins = histogram_by_alg(HSI, R, t, m, **model_dict)
                    TPR_list, FPR_list = calculate_TPR_FPR(NT_hist, WT_hist, bins)
                    auc = getAUC(TPR_list, FPR_list)
                    string = "{:.4f}".format(auc[0])
                    string = algorithm + "- dof{} | AUC = ".format(dof) + string
                    ax.plot(FPR_list, TPR_list, label=string)
                model_dict["degrees_of_freedom"] = dofs
            else:
                NT_hist, WT_hist, bins = histogram_by_alg(HSI, R, t, m, **model_dict)
                TPR_list, FPR_list = calculate_TPR_FPR(NT_hist, WT_hist, bins)
                string = algorithm
                ax.plot(FPR_list, TPR_list, label=string)
        ax.set_xlim([0, 0.1])
        ax.grid()
        ax.legend()
        string = "a/a_0 = " + "{:.2f}".format(ratio)
        ax.set_title("ROC Curve for all algorithms \n" + string, fontsize=14)
        ax.set_ylabel('TPR', fontsize=12)
        ax.set_xlabel('FPR', fontsize=12)
        plt.show()

    elif model_dict["simulation_type"] == "histogram_roc":
        fig, (axe1, axe2) = plt.subplots(1, 2)
        string = "a/a_0 = " + "{:.2f}".format(ratio)
        fig.suptitle(model_dict["algorithm"] + " - algorithm\n" + string)

        if is_tdist(model_dict["algorithm"]) and isinstance(model_dict["degrees_of_freedom"], list):
            dofs = model_dict["degrees_of_freedom"]
            for dof in dofs:
                model_dict["degrees_of_freedom"] = dof
                NT_hist, WT_hist, bins = histogram_by_alg(HSI, R, t, m, **model_dict)
                # plotting histograms
                string = "dof = {}  ".format(str(dof))
                axe1.plot(bins, NT_hist, label="no target  " + string)
                axe1.plot(bins, WT_hist, label="with target  " + string)
                # Plotting  ROC curve
                TPR_list, FPR_list = calculate_TPR_FPR(NT_hist, WT_hist, bins)
                auc = getAUC(TPR_list, FPR_list)

                axe2.plot(FPR_list, TPR_list, label=string + "AUC = " + "{:.5f}".format(auc))
        else:
            NT_hist, WT_hist, bins = histogram_by_alg(HSI, R, t, m, **model_dict)
            # plotting histograms
            axe1.plot(bins, NT_hist, color='b', label="no target  ")
            axe1.plot(bins, WT_hist, color='r', label="with target  ")
            # Plotting  ROC curve
            TPR_list, FPR_list = calculate_TPR_FPR(NT_hist, WT_hist, bins)
            auc = getAUC(TPR_list, FPR_list)
            string = "{:.4f}".format(auc[0])
            string = model_dict["algorithm"] + "\nAUC({}) = ".format(str(0.1)) + string + "\n"
            string = string + "AUC({}) = ".format(str(0.01)) + "{:.4f}".format(auc[1]) + "\n"
            string = string + "AUC({}) = ".format(str(0.001)) + "{:.5f}".format(auc[2])
            axe2.plot(FPR_list, TPR_list, label=string)

        axe1.set_yscale('log', nonposy='clip')
        axe1.legend()
        axe1.set_title("histograms")

        axe2.set_xlim([0, 0.01])
        # axe2.set_yscale('log')
        axe2.grid()
        axe2.legend()
        axe2.set_ylabel('TPR', fontsize=16)
        axe2.set_xlabel('FPR', fontsize=16)
        axe2.set_title("ROC")
        plt.show()

    elif model_dict["simulation_type"] == "Veritas-sigmas":
        sigmas = model_dict["sigmas"]
        results = np.zeros((3, len(sigmas)))

        fig, ax = plt.subplots()
        string = "a/a_0 = " + "{:.2f}".format(ratio)
        fig.suptitle(model_dict["algorithm"] + " \n" + string)
        ax.set_xlim([0, 0.01])
        ax.grid()
        sigmas = model_dict["sigmas"]
        for j,n in enumerate(sigmas):
            model_dict["sigmas"] = n
            NT_hist, WT_hist, bins = histogram_by_alg(HSI, R, t, m, **model_dict)
            TPR_list, FPR_list = calculate_TPR_FPR(NT_hist, WT_hist, bins)
            auc = getAUC(TPR_list, FPR_list)
            results[:, j] = np.asarray(auc)

        aa = ["AUC(0.1)","AUC(0.01)","AUC(0.001)"]
        for j,a in enumerate(aa):
            ax.plot(sigmas, results[j,:], label=a)
        ax.legend()
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_xlabel('n', fontsize=12)
        plt.show()


data_location = os.getcwd() + "\\Data\\self_test"

# algorithms = ["Veritas", "LMP", "Clairvoyant", "GLRT", "AMF", "ACE"]  # "EC_FTMF" ,
# "Clairvoyant replacement gaussian model"
# algorithms = ["LMP", "GLRT", "AMF", "ACE", "RX", "EC_FTMF",
#               "Clairvoyant additive multivariate-t model"
#               "Clairvoyant replacement gaussian model",
#               "Clairvoyant replacement multivariate-t model",
#               "Veritas additive multivariate-t model",
#               "Veritas replacement gaussian model",
#               "Veritas replacement multivariate-t model"]
# algorithms = ["LMP", "GLRT", "AMF", "ACE", "RX", "EC_FTMF",
#               "Clairvoyant additive multivariate-t model"
#               "Clairvoyant replacement gaussian model",
#               "Clairvoyant replacement multivariate-t model",
#               "Veritas additive multivariate-t model",
#               "Veritas replacement gaussian model",
#               "Veritas replacement multivariate-t model"]
algorithms = ["GLRT", "ACE","Clairvoyant additive multivariate-t model","Veritas additive multivariate-t model",
"Clairvoyant replacement multivariate-t model","Veritas replacement multivariate-t model"]

#               "Veritas replacement gaussian model",
#               "Veritas replacement multivariate-t model"]
#               ]
dof = [10, 928]
# list of targets and their corresponding location (y,x)
# F1 - (138, 504) a_0 = 0009456
# F2 - (122, 484) a_0 = 0.012916
# V1 - (128, 339) a_0 = 0.020590
# V2 - (156, 353) a_0 = 0.031596
# V3 - (186, 282) a_0 = 0.031731

sigmas = [ j for j in range(20)]

model_dict = {"target_name": "F1",
              "target_location": (122, 484),
              "target_strength": 0.05,
              "data_location": data_location,
              "algorithm": algorithms,
              # ["LMP", "GLRT", "AMF", "ACE", "RX", "Clairvoyant","EC_FTMF", "Veritas"]
              "recreate_statistical_model": False,  # false means we load our  pre calculated covariance and mean
              "simulation_type": "plot_all_roc",
              # "plot_all_roc" , "unknown_target_strength", "histogram_roc","Veritas-sigmas"
              "degrees_of_freedom": 10,
              "sigmas": 4
              }

simulate(**model_dict)
# for alg in algorithms:
#     model_dict["degrees_of_freedom"] = dof
#     model_dict["algorithm"] = alg
#     simulate(**model_dict)


