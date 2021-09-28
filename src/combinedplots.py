import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

#adds CAI to results
def addCAI(results,acc,accgap,alpha):

    # accgap_baseline - accgap_debiased                 acc_debiased - acc_baseline
    CAI = alpha * (accgap - results[:,1]) + (1- alpha) * (results[:,0] - acc)

    return np.column_stack((results,CAI))

'0,1,2,3 - acc, accgap, dpgap, eqoddsgap'
metrics = ["acc","accgap","dpgap","eqoddsgap"]
datasets = ["CelebA_gender", "CelebA_race", "EyePACS", "Adult", 'Mh_age', 'Mh_gender', 'fairface','fairface_race']
methods = ["IB", "Skoglund", "Combined"]


dataset_type = 1


# results_A =  np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_alpha.npy')
# results_B =  np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_beta.npy')

#eyepacs combined code



# #

#

baseline_acc = [0.6908,0.7444,0.70,-1,-1,-1,-1,-1]
baseline_acc_gap = [0.2165,0.1393,0.035,-1,-1,-1,-1,-1]


alpha = 0.75

method = 2

if dataset_type == 2: #eyepacs
    alphas_2_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_1.npy')
    results_2_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_1.npy')

    alphas_2_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_2.npy')
    results_2_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_2.npy')

    alphas_2 = np.append(alphas_2_1,alphas_2_2)
    results_2 = np.append(results_2_1,results_2_2,axis=0)

else:
    alphas_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas.npy')
    results_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results.npy')
results_2 = addCAI(results_2,baseline_acc[dataset_type],baseline_acc_gap[dataset_type],alpha)


method = 1
alphas_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas.npy')
results_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results.npy')
results_1 = addCAI(results_1,baseline_acc[dataset_type],baseline_acc_gap[dataset_type],alpha)


method = 0
alphas_0 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas.npy')
results_0 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results.npy')
results_0 = addCAI(results_0,baseline_acc[dataset_type],baseline_acc_gap[dataset_type],alpha)


save = True


#
def generate_bar2(metric,title,ylabel,ending,A,B,N,alphas):

    ind = np.arange(N)
    width = 0.2

    distance = round(B - A,2)
    increment = distance / N


    Ycombined =[]
    Yskoglund = []
    Yib = []

    for i in range(N):
        combined = []
        skoglund = []
        ib = []
        start = A + increment *i

        if alphas:
            for j in range(len(alphas_0)):
                if alphas_0[j] >= start and alphas_0[j] < start + increment:
                    ib.append(results_0[j,metric]*100)
            for j in range(len(alphas_1)):
                if alphas_1[j] >= start and alphas_1[j] < start + increment:
                    skoglund.append(results_1[j,metric]*100)
            for j in range(len(alphas_2)):
                if alphas_2[j] >= start and alphas_2[j] < start + increment:
                    combined.append(results_2[j,metric]*100)
        else:
            for j in range(len(alphas_0)):
                if results_0[j,0] >= start and results_0[j,0] < start + increment:
                    ib.append(results_0[j,metric]*100)
            for j in range(len(alphas_1)):
                if results_1[j,0] >= start and results_1[j,0] < start + increment:
                    skoglund.append(results_1[j,metric]*100)
            for j in range(len(alphas_2)):
                if results_2[j,0] >= start and results_2[j,0] < start + increment:
                    combined.append(results_2[j,metric]*100)
        if np.mean(ib) is not None:
            Yib.append(np.mean(ib))
        else:
            Yib.append(0)

        if np.mean(skoglund) is not None:
            Yskoglund.append(np.mean(skoglund))
        else:
            Yskoglund.append(0)

        if np.mean(combined) is not None:
            Ycombined.append(np.mean(combined))
        else:
            Ycombined.append(0)


    X = []
    if alphas:
        q = f'{A}-{A+increment}'
        X.append(q)
        w = f'{A+increment:.2f}-{A+increment*2:.2f}'
        X.append(w)
        e = f'{A + increment*2:.2f}-{A + increment * 3:.2f}'
        X.append(e)
        r = f'{A + increment * 3:.2f}-{A + increment * 4:.2f}'
        X.append(r)
        if N == 5:
            t = f'{A + increment * 4:.2f}-{A + increment * 5:.2f}'
            X.append(t)
    else:
        q = f'{A*100:.2f}-{(A+increment)*100:.2f}'
        X.append(q)
        w = f'{(A+increment)*100:.2f}-{(A+increment*2)*100:.2f}'
        X.append(w)
        e = f'{(A + increment*2)*100:.2f}-{(A + increment * 3)*100:.2f}'
        X.append(e)
        r = f'{(A + increment * 3)*100:.2f}-{(A + increment * 4)*100:.2f}'
        X.append(r)
        if N >= 5:
            t = f'{(A + increment * 4)*100:.2f}-{(A + increment * 5)*100:.2f}'
            X.append(t)
        if N == 6:
            y = f'{(A + increment * 5)*100:.2f}-{(A + increment * 6)*100:.2f}'
            X.append(y)

    my_cmap = plt.get_cmap("viridis")

    plt.bar(ind,Ycombined,width,label = 'Combined',color=my_cmap.colors)
    plt.bar(ind+width,Yskoglund,width,label = 'Skoglund')
    plt.bar(ind+width*2,Yib,width,label = 'IB')


    plt.title(title)
    if alphas:
        plt.xlabel(r"$\alpha$")
    else:
        plt.xlabel("accuracy")
    plt.ylabel(ylabel)
    # plt.gca().invert_xaxis()
    plt.xticks(ind+width,X)
    plt.grid(True)

    plt.legend()
    if save:
        plt.savefig(f"../../newestplots2/{datasets[dataset_type]}_{ending}_bar.png")
    plt.show()


def plot_fullscatter(metric,title,ylabel,alphas):
    plt.title(title)
    plt.ylabel(ylabel)

    if alphas:
        plt.xlabel(r"$\alpha$")
        # plt.gca().invert_xaxis()
        plt.plot(alphas_2, results_2[:,metric]*100,'*',label='Combined')
        plt.plot(alphas_1, results_1[:,metric]*100,'*',label='Skoglund')
        plt.plot(alphas_0, results_0[:,metric]*100,'*',label='IB')
    else:
        plt.xlabel("accuracy")
        plt.plot(results_2[:, 0]*100, results_2[:, metric]*100, '*', label='Combined')
        plt.plot(results_1[:, 0]*100, results_1[:, metric]*100, '*', label='Skoglund')
        plt.plot(results_0[:, 0]*100, results_0[:, metric]*100, '*', label='IB')

    plt.legend()
    if save:
        plt.savefig(f"../../newestplots2/{datasets[dataset_type]}_{ending}.png")
    plt.show()



'''Alpha plots'''
# #

title = rf"{datasets[dataset_type]} CAI vs $\alpha$"
ylabel = 'CAI'
ending = 'CAI'
plot_fullscatter(6,title,ylabel,alphas=True)
generate_bar2(6,title,ylabel,ending,0,0.5,N=5,alphas=True)


title = rf"{datasets[dataset_type]} Accuracy vs $\alpha$"
ylabel = 'Accuracy'
ending = 'acc_alpha'
plot_fullscatter(0,title,ylabel,alphas=True)
generate_bar2(0,title,ylabel,ending,0,0.5,N=5,alphas=True)


title = rf"{datasets[dataset_type]} Accuracy Gap vs $\alpha$"
ylabel = 'Acc Gap'
ending = 'accgap_alpha'
plot_fullscatter(1,title,ylabel,alphas=True)
generate_bar2(1,title,ylabel,ending,0,0.5,N=5,alphas=True)

title = rf"{datasets[dataset_type]} Discrimination vs $\alpha$"
ylabel = "dp gap"
ending = "disc_alpha"
plot_fullscatter(2,title,ylabel,alphas=True)
generate_bar2(2,title,ylabel,ending,0,0.5,N=5,alphas=True)

title = rf"{datasets[dataset_type]} Equalized Odds vs $\alpha$"
ylabel = "eq odds gap"
ending = "eqodds_alpha"
plot_fullscatter(3,title,ylabel,alphas=True)
generate_bar2(3,title,ylabel,ending,0,0.5,N=5,alphas=True)


# ''' vs Accuracy Comparisons  '''

title = rf"{datasets[dataset_type]} Acc Gap vs Accuracy"
ylabel = "acc gap"
ending = "accgap_acc"
plot_fullscatter(1,title,ylabel,alphas=False)
generate_bar2(1,title,ylabel,ending,0.68,0.75,N=5,alphas=False)


title = rf"{datasets[dataset_type]} Discrimination vs Accuracy"
ylabel = 'dp gap'
ending = "disc_acc"
plot_fullscatter(2,title,ylabel,alphas=False)
generate_bar2(2,title,ylabel,ending,0.68,0.75,N=5,alphas=False)

title = rf"{datasets[dataset_type]} Equalized Odds vs Accuracy"
ylabel = 'eq odds gap'
ending = "eqodds_acc"
plot_fullscatter(3,title,ylabel,alphas=False)
generate_bar2(3,title,ylabel,ending,0.68,0.75,N=5,alphas=False)
#































#
# plt.title(rf"{datasets[dataset_type]} Discrimination v Accuracy")
# plt.ylabel("dp gap")
# plt.xlabel("accuracy")
# plt.plot(results_B[:,0], results_B[:,2],'*',label="Skoglund orignal")
# plt.plot(results_A[:,0], results_A[:,2],'*',label='Skoglund Renyi divergence')
# # plt.plot(results_0[:,0], results_0[:,2],'*',label='IB')
# # plt.gca().invert_xaxis()
# plt.legend()
# if save:
#     plt.savefig(f"../../newestplots/{datasets[dataset_type]}_dp_acc.png")
# plt.show()
#
# plt.title(rf"{datasets[dataset_type]} Accuracy Gap vs Accuracy")
# plt.ylabel("acc gap")
# plt.xlabel("accuracy")
# plt.plot(results_B[:,0], results_B[:,1],'*',label="Skoglund orignal")
# plt.plot(results_A[:,0], results_A[:,1],'*',label='Skoglund alpha')
# # plt.plot(results_1[:,0], results_1[:,1],'*',label='Skoglund')
# # plt.plot(results_0[:,0], results_0[:,1],'*',label='IB')
# # plt.gca().invert_xaxis()
# plt.legend()
# if save:
#     plt.savefig(f"../../newestplots/{datasets[dataset_type]}_accgap_acc.png")
# plt.show()


# plt.plot(results[:,0],results[:,2],'*')

# plt.plot(alphas, results[:,1],'*')
# plt.plot(alphas, results[:,2],'*')
# plt.plot(alphas, results[:,3],'*')




# plt.legend()
# # plt.savefig("../../plots/celeba_race_trainloss.png")
# plt.show()




#old generatebar function
def generate_bar(metric,title,ylabel,ending):
    N=5
    ind = np.arange(N)
    width = 0.2

    Ycombined =[]
    Yskoglund = []
    Yib = []

    for i in range(5):
        combined = []
        skoglund = []
        ib = []
        start = 0.2 *i

        for j in range(len(alphas_0)):
            if alphas_0[j] >= start and alphas_0[j] < start + 0.2:
                ib.append(results_0[j,metric])
        for j in range(len(alphas_1)):
            if alphas_1[j] >= start and alphas_1[j] < start + 0.2:
                skoglund.append(results_1[j,metric])
        for j in range(len(alphas_2)):
            if alphas_2[j] >= start and alphas_2[j] < start + 0.2:
                combined.append(results_2[j,metric])
        Yib.append(np.mean(ib))

        Yskoglund.append(np.mean(skoglund))
        Ycombined.append(np.mean(combined))

    X = ['0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1']
    plt.bar(ind,Ycombined,width,label = 'Combined')
    plt.bar(ind+width,Yskoglund,width,label = 'Skoglund')
    plt.bar(ind+width*2,Yib,width,label = 'IB')


    plt.title(title)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(ylabel)
    # plt.gca().invert_xaxis()
    plt.xticks(ind+width,X)
    plt.grid(True)

    plt.legend()
    if save:
        plt.savefig(f"../../newestplots2/{datasets[dataset_type]}_{ending}")
    plt.show()