import numpy as np
import matplotlib.pyplot as plt

'0,1,2,3 - acc, accgap, dpgap, eqoddsgap'

datasets = ["CelebA_gender", "CelebA_race", "EyePACS", "Adult", 'Mh_age', 'Mh_gender']
methods = ["IB", "Skoglund", "Combined"]


dataset_type = 3
method = 1

results_A =  np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_alpha.npy')
results_B =  np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_beta.npy')



# alphas_2_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_1.npy')
# results_2_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_1.npy')
#
#
# alphas_2_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_2.npy')
# results_2_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_2.npy')
#
# alphas_2 = np.append(alphas_2_1,alphas_2_2)
# results_2 = np.append(results_2_1,results_2_2,axis=0)

# method = 1
# alphas_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas.npy')
# results_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results.npy')
#
# method = 0
# alphas_0 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas.npy')
# results_0 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results.npy')

save = True

# # alphas = np.log(alphas)
# plt.title(rf"{datasets[dataset_type]} Accuracy vs $\alpha$")
# plt.xlabel(r"$\alpha$")
# plt.ylabel("Accuracy")
# # plt.gca().invert_xaxis()
# plt.plot(alphas_2, results_2[:,0],'*',label='Combined')
# # plt.plot(alphas_1, results_1[:,0],'*',label='Skoglund')
# # plt.plot(alphas_0, results_0[:,0],'*',label='IB')
#
# plt.legend()
# if save:
#     plt.savefig(f"../../newestplots/{datasets[dataset_type]}_acc_alpha.png")
# plt.show()
#
# plt.title(rf"{datasets[dataset_type]} Accuracy Gap vs $\alpha$")
# plt.xlabel(r"$\alpha$")
# plt.ylabel("acc gap")
# plt.plot(alphas_2, results_2[:,1],'*',label='Combined')
# # plt.plot(alphas_1, results_1[:,1],'*',label='Skoglund')
# # plt.plot(alphas_0, results_0[:,1],'*',label='IB')
#
# # plt.gca().invert_xaxis()
# plt.legend()
# if save:
#     plt.savefig(f"../../newestplots/{datasets[dataset_type]}_accgap_alpha.png")
# plt.show()
#
# plt.title(rf"{datasets[dataset_type]} Discrimination vs $\alpha$")
# plt.xlabel(r"$\alpha$")
# plt.ylabel("dp gap")
# plt.plot(alphas_2, results_2[:,2],'*',label='Combined')
# # plt.plot(alphas_1, results_1[:,2],'*',label='Skoglund')
# # plt.plot(alphas_0, results_0[:,2],'*',label='IB')
# # plt.gca().invert_xaxis()
# plt.legend()
# if save:
#     plt.savefig(f"../../newestplots/{datasets[dataset_type]}_dp_alpha.png")
# plt.show()
#
# plt.title(rf"{datasets[dataset_type]} Equalized Odds vs $\alpha$")
# plt.xlabel(r"$\alpha$")
# plt.ylabel("eq odds gap")
# plt.plot(alphas_2, results_2[:,3],'*',label='Combined')
# # plt.plot(alphas_1, results_1[:,3],'*',label='Skoglund')
# # plt.plot(alphas_0, results_0[:,3],'*',label='IB')
# # plt.gca().invert_xaxis()
# plt.legend()
# if save:
#     plt.savefig(f"../../newestplots/{datasets[dataset_type]}_eqodds_alpha")
# plt.show()
#
#
plt.title(rf"{datasets[dataset_type]} Equalized Odds vs Accuracy")
plt.ylabel("eq odds gap")
plt.xlabel("accuracy")
plt.plot(results_B[:,0], results_B[:,3],'*',label="Skoglund orignal")
plt.plot(results_A[:,0], results_A[:,3],'*',label='Skoglund alpha')
# plt.plot(results_2[:,0], results_2[:,3],'*',label='Combined')
# plt.plot(results_1[:,0], results_1[:,3],'*',label='Skoglund')
# plt.plot(results_0[:,0], results_0[:,3],'*',label='IB')
# plt.gca().invert_xaxis()
plt.legend()
if save:
    plt.savefig(f"../../newestplots/{datasets[dataset_type]}_eqodds_acc.png")
plt.show()
#
plt.title(rf"{datasets[dataset_type]} Discrimination v Accuracy")
plt.ylabel("dp gap")
plt.xlabel("accuracy")
plt.plot(results_B[:,0], results_B[:,2],'*',label="Skoglund orignal")
plt.plot(results_A[:,0], results_A[:,2],'*',label='Skoglund Renyi divergence')
# plt.plot(results_0[:,0], results_0[:,2],'*',label='IB')
# plt.gca().invert_xaxis()
plt.legend()
if save:
    plt.savefig(f"../../newestplots/{datasets[dataset_type]}_dp_acc.png")
plt.show()

plt.title(rf"{datasets[dataset_type]} Accuracy Gap vs Accuracy")
plt.ylabel("acc gap")
plt.xlabel("accuracy")
plt.plot(results_B[:,0], results_B[:,1],'*',label="Skoglund orignal")
plt.plot(results_A[:,0], results_A[:,1],'*',label='Skoglund alpha')
# plt.plot(results_1[:,0], results_1[:,1],'*',label='Skoglund')
# plt.plot(results_0[:,0], results_0[:,1],'*',label='IB')
# plt.gca().invert_xaxis()
plt.legend()
if save:
    plt.savefig(f"../../newestplots/{datasets[dataset_type]}_accgap_acc.png")
plt.show()


# plt.plot(results[:,0],results[:,2],'*')

# plt.plot(alphas, results[:,1],'*')
# plt.plot(alphas, results[:,2],'*')
# plt.plot(alphas, results[:,3],'*')




# plt.legend()
# # plt.savefig("../../plots/celeba_race_trainloss.png")
# plt.show()