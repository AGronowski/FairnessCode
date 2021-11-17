import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


def quickCAI(alpha,acc,accgap,acc_b,accgap_b):
    CAI = alpha * (accgap_b- accgap) + (1- alpha) * (acc - acc_b)
    print(CAI)
    if alpha == 0.5:
        quickCAI(.75,acc,accgap,acc_b,accgap_b)

quickCAI(0.5,70.83,1.33,73.37,8.08)

#adds CAI to results
def addCAI(results,acc,accgap,alpha):

    # accgap_baseline - accgap_debiased                 acc_debiased - acc_baseline
    CAI = alpha * (accgap/100 - results[:,1]) + (1- alpha) * (results[:,0] - acc/100)

    return np.column_stack((results,CAI))

'0,1,2,3,4,5,6 - acc, accgap, dpgap, eqoddsgap, accmin0, accmin1, CAI'
metrics = ["acc","accgap","dpgap","eqoddsgap"]
# 0 - 8
datasets = ["CelebA_gender", "CelebA_race", "EyePACS", "Adult", 'Mh_age', 'Mh_gender', 'fairface','fairface_race','mnist']
methods = ["IB", "Skoglund", "Combined"]


dataset_type = 0
CAI_lambda = 0.75
b1orb2 = 'b1'



# results_A =  np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_alpha.npy')
# results_B =  np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_beta.npy')

#eyepacs combined code



# #

#

# baseline_acc = [0.6908,0.7444,0.70,-1,-1,-1,-1,-1,-1]
# baseline_acc_gap = [0.2165,0.1393,0.035,-1,-1,-1,-1,-1,-1]

#order matches datasets. celeba gender, race, eyepacs, ...
baseline_acc = [69.025,70.61,73.37,-1,-1,-1,-1,-1,-1]
baseline_acc_gap = [20.25,16.57,08.08,-1,-1,-1,-1,-1,-1]
baseline_dp_gap = [45.60,43.82,28.25,-1,-1,-1,-1,-1,-1]
baseline_eqodds_gap = [65.85,60.4,36.33,-1,-1,-1,-1,-1,-1]
baseline_accmin0 = [79.15,78.9,77.41,-1,-1,-1,-1,-1,-1]
baseline_accmin1 = [58.9,62.3,69.33,-1,-1,-1,-1,-1,-1]

ib_eyepacs = [74.12,2.02,18.58,20.67,73.08,3.37,4.69]
skoglund_eyepacs = [77.83,1.66,10.83,12.5,77,5.84,5.93]



if dataset_type == 20: #never run
    method = 2
    alphas_2_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_1.npy')
    results_2_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_1.npy')

    alphas_2_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_2.npy')
    results_2_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_results_2.npy')

    alphas_2 = np.append(alphas_2_1,alphas_2_2)
    results_2 = np.append(results_2_1,results_2_2,axis=0)

else:
    method = 2
    alphas_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_alphas.npy')
    results_2 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_results.npy')
    beta1s = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_beta1s.npy')
    beta2s = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_beta2s.npy')
    betas = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_betas.npy')
results_2 = addCAI(results_2, baseline_acc[dataset_type], baseline_acc_gap[dataset_type], .5)
results_2 = addCAI(results_2, baseline_acc[dataset_type], baseline_acc_gap[dataset_type], .75)


method = 1
alphas_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_b_alphas.npy')
results_1 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_b_results.npy')
results_1 = addCAI(results_1, baseline_acc[dataset_type], baseline_acc_gap[dataset_type], CAI_lambda)


method = 0
alphas_0 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_b_alphas.npy')
results_0 = np.load(f'../results/{datasets[dataset_type]}_{methods[method]}_b_results.npy')
results_0 = addCAI(results_0, baseline_acc[dataset_type], baseline_acc_gap[dataset_type], CAI_lambda)


save = True
plot_bar = True
plot_scatter = False


#
def generate_bar2(metric,title,ylabel,ending,A,B,N,alphas,baseline=None):

    plt.clf()
    ind = np.arange(N)
    width = 0.2

    distance = round(B - A,2)
    increment = distance / N


    Ycombined =[]
    Yskoglund = []
    Yib = []


    xaxis = []

    for i in range(N):
        combined = []
        skoglund = []
        ib = []
        start = A + increment *i
        xaxis.append(start*100)

        if alphas:
            # for j in range(len(alphas_0)):
            #     if alphas_0[j] >= start and alphas_0[j] < start + increment:
            #         ib.append(results_0[j,metric]*100)
            # for j in range(len(alphas_1)):
            #     if alphas_1[j] >= start and alphas_1[j] < start + increment:
            #         skoglund.append(results_1[j,metric]*100)
            for j in range(len(alphas_2)):
                if alphas_2[j] >= start and alphas_2[j] < start + increment:
                    combined.append(results_2[j,metric]*100)
        else:
            # for j in range(len(alphas_0)):
            #     if results_0[j,0] >= start and results_0[j,0] < start + increment:
            #         ib.append(results_0[j,metric]*100)
            # for j in range(len(alphas_1)):
            #     if results_1[j,0] >= start and results_1[j,0] < start + increment:
            #         skoglund.append(results_1[j,metric]*100)
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
            Yskoglund.append(np.nan)

        if len(combined) > 0:
            Ycombined.append(np.min(combined))
        else:
            Ycombined.append(np.nan)


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
        if N >= 5:
            t = f'{A + increment * 4:.2f}-{A + increment * 5:.2f}'
            X.append(t)
        if N == 6:
            y = f'{(A + increment * 5)*100:.2f}-{(A + increment * 6)*100:.2f}'
            X.append(y)
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

    plt.plot(xaxis,Ycombined,'--o',label = 'Combined')
    # plt.bar(ind+width,Yskoglund,width,label = 'Skoglund')
    # plt.bar(ind+width*2,Yib,width,label = 'IB')
    # array = [0, 0, 0, 0, baseline]
    # if baseline is not None:
    #     plt.bar(ind+width*3,array,width,label= "Baseline")


    plt.title(title)
    if alphas:
        plt.xlabel(r"$\alpha$")
    else:
        plt.xlabel("accuracy")
    plt.ylabel(ylabel)
    # plt.gca().invert_xaxis()

    # X = [0,0.2,0.4,0.6,0.8,1]

    # plt.xticks(xaxis)
    plt.grid(True)

    # plt.legend()
    if plot_bar:
        if save:
            plt.savefig(f"../../newestplots2/{datasets[dataset_type]}_{ending}_bar.png")
        plt.show()


def plot_fullscatter(metric,title,ylabel,xlabel,xaxis,baseline,ib,skoglund,alphas_or_betas):
    plt.clf()
    plt.title(title)
    plt.ylabel(ylabel)


    if alphas_or_betas:
        plt.xlabel(xlabel)
        # plt.gca().invert_xaxis()
        plt.plot(xaxis, results_2[:,metric]*100,'*',label='Combined')
        # plt.plot(alphas_1, results_1[:,metric]*100,'*',label='Skoglund')
        # plt.plot(alphas_0, results_0[:,metric]*100,'*',label='IB')

    else:
        plt.xlabel("accuracy")
        plt.plot(results_2[:, 0]*100, results_2[:, metric]*100, '*', label='Combined')
        # plt.plot(results_1[:, 0]*100, results_1[:, metric]*100, '*', label='Skoglund')
        # plt.plot(results_0[:, 0]*100, results_0[:, metric]*100, '*', label='IB')
    if baseline != None:
        plt.axhline(y=baseline, color='b', linestyle='-',label="Baseline")

#temporary fix for data with no accmin and accmax
    if metric == 6:
        metric = 4
    plt.axhline(skoglund, color='r', linestyle='-',label="Skoglund")
    plt.axhline(ib, color='g', linestyle='-',label="IB")


    plt.plot()

    plt.legend()

    if plot_scatter:
        if save:
            plt.savefig(f"../../newestplots2/{datasets[dataset_type]}_{ending}.png")
        plt.show()



'''Alpha plots'''
# #

show_alpha = False
if b1orb2 == 'b1':
    b1 = True #beta1 or beta2
    beta1or2 = 'beta1'
else:
    b1 = False
    beta1or2 = 'beta2'
if show_alpha:

    '''CAI '''
    title = rf"{datasets[dataset_type]} CAI $\lambda=${CAI_lambda} vs $\alpha$"
    ylabel = 'CAI'
    xlabel = r"$\alpha$"
    ending = 'CAI'
    plot_fullscatter(6,title,ylabel,xlabel,alphas_2,0,0,0,alphas_or_betas=True)
    generate_bar2(6,title,ylabel,ending,0,1,N=6,alphas=True)

    title = rf"{datasets[dataset_type]} CAI $\alpha=${CAI_lambda} vs $\{beta1or2}$"
    ylabel = 'CAI'
    ending = f'CAI_{beta1or2}'
    if b1:
        xlabel = r'$\beta1$'
        plot_fullscatter(6,title,ylabel,xlabel,beta1s,0,0,0,alphas_or_betas=True)
    else:
        xlabel = r'$\beta2$'
        plot_fullscatter(6,title,ylabel,xlabel,beta2s,0,0,0,alphas_or_betas=True)

    '''Accuracy'''

    title = rf"{datasets[dataset_type]} Accuracy vs $\alpha$"
    ylabel = 'Accuracy'
    ending = 'acc_alpha'
    xlabel = r"$\alpha$"
    plot_fullscatter(0,title,ylabel,xlabel,alphas_2,baseline_acc[dataset_type],ib_eyepacs[0],skoglund_eyepacs[0],alphas_or_betas=True)
    generate_bar2(0,title,ylabel,ending,0,1,N=6,alphas=True)

    title = rf"{datasets[dataset_type]} Accuracy vs $\{beta1or2}$"
    ylabel = 'Accuracy'
    xlabel = r'$\beta1$'
    ending = f'acc_{beta1or2}'
    if b1:
        xlabel = r'$\beta1$'
        plot_fullscatter(0,title,ylabel,xlabel,beta1s,baseline_acc[dataset_type],ib_eyepacs[0],skoglund_eyepacs[0],alphas_or_betas=True)
    else:
        xlabel = r'$\beta2$'
        plot_fullscatter(0,title,ylabel,xlabel,beta2s,baseline_acc[dataset_type],ib_eyepacs[0],skoglund_eyepacs[0],alphas_or_betas=True)
    # generate_bar2(0,title,ylabel,ending,0,1,N=6,alphas=True)

    '''Acc gap '''

    title = rf"{datasets[dataset_type]} Accuracy Gap vs $\alpha$"
    ylabel = 'Acc Gap'
    xlabel = r'$\alpha$'
    ending = 'accgap_alpha'
    plot_fullscatter(1,title,ylabel,xlabel,alphas_2,baseline_acc_gap[dataset_type],ib_eyepacs[1],skoglund_eyepacs[1],alphas_or_betas=True)
    generate_bar2(1,title,ylabel,ending,0,1,N=6,alphas=True)

    title = rf"{datasets[dataset_type]} Accuracy Gap vs $\{beta1or2}$"
    ylabel = 'Acc Gap'
    xlabel = r'$\beta1$'
    ending = f'accgap_{beta1or2}'
    if b1:
        xlabel = r'$\beta1$'
        plot_fullscatter(1,title,ylabel,xlabel,beta1s,baseline_acc_gap[dataset_type],ib_eyepacs[1],skoglund_eyepacs[1],alphas_or_betas=True)
    else:
        xlabel = r'$\beta2$'
        plot_fullscatter(1,title,ylabel,xlabel,beta2s,baseline_acc_gap[dataset_type],ib_eyepacs[1],skoglund_eyepacs[1],alphas_or_betas=True)

    '''DP gap'''

    title = rf"{datasets[dataset_type]} Discrimination vs $\alpha$"
    ylabel = "dp gap"
    xlabel = r'$\alpha$'
    ending = "disc_alpha"
    plot_fullscatter(2,title,ylabel,xlabel,alphas_2,baseline_dp_gap[dataset_type],ib_eyepacs[2],skoglund_eyepacs[2],alphas_or_betas=True)
    generate_bar2(2,title,ylabel,ending,0,1,N=6,alphas=True)

    title = rf"{datasets[dataset_type]} Discrimination vs $\{beta1or2}$"
    ylabel = "dp gap"
    xlabel = fr'$\{beta1or2}$'
    ending = f"disc_{beta1or2}"
    if b1:
        xlabel = r'$\beta1$'
        plot_fullscatter(2,title,ylabel,xlabel,beta1s,baseline_dp_gap[dataset_type],ib_eyepacs[2],skoglund_eyepacs[2],alphas_or_betas=True)
    else:
        xlabel = r'$\beta2$'
        plot_fullscatter(2,title,ylabel,xlabel,beta2s,baseline_dp_gap[dataset_type],ib_eyepacs[2],skoglund_eyepacs[2],alphas_or_betas=True)
    '''Eq odds gap'''

    title = rf"{datasets[dataset_type]} Equalized Odds vs $\alpha$"
    ylabel = "eq odds gap"
    xlabel = r'$\alpha$'
    ending = "eqodds_alpha"
    plot_fullscatter(3,title,ylabel,xlabel,alphas_2,baseline_eqodds_gap[dataset_type],ib_eyepacs[3],skoglund_eyepacs[3],alphas_or_betas=True)
    generate_bar2(3,title,ylabel,ending,0,1,N=6,alphas=True)

    title = rf"{datasets[dataset_type]} Equalized Odds vs $\{beta1or2}$"
    ylabel = "eq odds gap"
    xlabel = r'$\beta1$'
    ending = f"eqodds_{beta1or2}"
    if b1:
        xlabel = r'$\beta1$'
        plot_fullscatter(3,title,ylabel,xlabel,beta1s,baseline_eqodds_gap[dataset_type],ib_eyepacs[3],skoglund_eyepacs[3],alphas_or_betas=True)
    else:
        xlabel = r'$\beta2$'
        plot_fullscatter(3,title,ylabel,xlabel,beta2s,baseline_eqodds_gap[dataset_type],ib_eyepacs[3],skoglund_eyepacs[3],alphas_or_betas=True)
    # generate_bar2(3,title,ylabel,ending,0,1,N=6,alphas=True)

# ''' vs Accuracy Comparisons  '''

# def plot_fullscatter(metric,title,ylabel,xlabel,xaxis,baseline,alphas_or_betas):

plot_scatter = True
title = rf"{datasets[dataset_type]} Acc Gap vs Accuracy"
ylabel = "acc gap"

ending = "accgap_acc"
plot_fullscatter(1,title,ylabel,'',None,baseline_acc_gap[dataset_type],ib_eyepacs[1],skoglund_eyepacs[1],alphas_or_betas=False)
# generate_bar2(1,title,ylabel,ending,0.70,0.82,N=12,alphas=False,baseline=baseline_acc_gap[dataset_type])
generate_bar2(1,title,ylabel,ending,0.72,0.80,N=40,alphas=False,baseline=baseline_acc_gap[dataset_type])


title = rf"{datasets[dataset_type]} Discrimination vs Accuracy"
ylabel = 'dp gap'
ending = "disc_acc"
plot_fullscatter(2,title,ylabel,'',None,baseline_dp_gap[dataset_type],ib_eyepacs[2],skoglund_eyepacs[2],alphas_or_betas=False)
generate_bar2(2,title,ylabel,ending,0.68,0.82,N=60,alphas=False,baseline=baseline_dp_gap[dataset_type])

title = rf"{datasets[dataset_type]} Equalized Odds vs Accuracy"
ylabel = 'eq odds gap'
ending = "eqodds_acc"
plot_fullscatter(3,title,ylabel,'',None,baseline_dp_gap[dataset_type],ib_eyepacs[3],skoglund_eyepacs[3],alphas_or_betas=False)
generate_bar2(3,title,ylabel,ending,0.72,0.80,N=40,alphas=False,baseline=baseline_eqodds_gap[dataset_type])
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