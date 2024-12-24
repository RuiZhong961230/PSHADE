from copy import deepcopy
import os
import numpy as np
from cec17_functions import cec17_test_func
from scipy.stats import cauchy


PopSize = 50
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 10
MaxFEs = 1000 * DimSize
curFEs = 0
Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0

H = 5
muF1, muF2 = [0.5] * H, [0.5] * H
muCr1, muCr2 = [0.5] * H, [0.5] * H


def func(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


def meanWL(S, Delta):
    numer = 0
    denom = 0
    SumDelta = sum(Delta)
    for i in range(len(S)):
        numer += (Delta[i] / SumDelta) * S[i] ** 2
        denom += (Delta[i] / SumDelta) * S[i]
    return numer / denom


def meanWA(S, Delta):
    res = 0
    SumDelta = sum(Delta)
    for i in range(len(S)):
        res += (Delta[i] / SumDelta) * S[i]
    return res


def Initialization():
    global Pop, FitPop, curFEs, DimSize, muF1, muF2, muCr1, muCr2, H
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
        curFEs += 1
    muF1, muF2 = [0.5] * H, [0.5] * H
    muCr1, muCr2 = [0.5] * H, [0.5] * H


def PSHADE():
    global Pop, FitPop, LB, UB, PopSize, DimSize, curFEs, muF1, muF2, muCr1, muCr2
    F1_List, F2_List, Cr1_List, Cr2_List = [], [], [], []
    Delta_1, Delta_2 = [], []
    c = 0.1
    sigma = 0.1
    P1 = int(PopSize * (1 - (curFEs / MaxFEs)))
    F1_idx, F2_idx, Cr1_idx, Cr2_idx = np.random.randint(0, H, 4)
    for i in range(P1):  # DE/cur-to-best/1
        best_idx = np.argmin(FitPop)
        candi = list(range(PopSize))
        candi.remove(i)
        if i != best_idx:
            candi.remove(best_idx)
        r1, r2 = np.random.choice(candi, 2, replace=False)
        
        F = cauchy.rvs(muF1[F1_idx], sigma)
        while True:
            if F > 1:
                F = 1
                break
            elif F < 0:
                F = cauchy.rvs(muF1[F1_idx], sigma)
            break

        Off = Pop[i] + F * (Pop[best_idx] - Pop[i]) + F * (Pop[r1] - Pop[r2])

        jrand = np.random.randint(0, DimSize)
        Cr = np.clip(np.random.normal(muCr1[Cr1_idx], sigma), 0, 1)
        for j in range(DimSize):
            if np.random.rand() < Cr or j == jrand:
                pass
            else:
                Off[j] = Pop[i][j]

            if Off[j] > UB[j] or Off[j] < LB[j]:
                Off[j] = np.random.uniform(LB[j], UB[j])

        FitOff = func(Off)
        curFEs += 1
        if FitOff < FitPop[i]:
            F1_List.append(F)
            Cr1_List.append(Cr)
            Delta_1.append(abs(FitOff - FitPop[i]))
            Pop[i] = deepcopy(Off)
            FitPop[i] = FitOff

    for i in range(P1, PopSize):  # DE/best/1
        best_idx = np.argmin(FitPop)
        candi = list(range(PopSize))
        candi.remove(i)
        if i != best_idx:
            candi.remove(best_idx)
        r1, r2 = np.random.choice(candi, 2, replace=False)

        F = cauchy.rvs(muF2[F2_idx], sigma)
        while True:
            if F > 1:
                F = 1
                break
            elif F < 0:
                F = cauchy.rvs(muF2[F2_idx], sigma)
            break

        Off = Pop[best_idx] + F * (Pop[r1] - Pop[r2])

        jrand = np.random.randint(0, DimSize)
        Cr = np.clip(np.random.normal(muCr2[Cr2_idx], sigma), 0, 1)
        for j in range(DimSize):
            if np.random.rand() < Cr or j == jrand:
                pass
            else:
                Off[j] = Pop[i][j]

            if Off[j] > UB[j] or Off[j] < LB[j]:
                Off[j] = np.random.uniform(LB[j], UB[j])

        FitOff = func(Off)
        curFEs += 1
        if FitOff < FitPop[i]:
            F2_List.append(F)
            Cr2_List.append(Cr)
            Delta_2.append(abs(FitOff - FitPop[i]))
            Pop[i] = deepcopy(Off)
            FitPop[i] = FitOff

    if len(F1_List) == 0:
        pass
    else:
        muF1[F1_idx] = (1 - c) * muF1[F1_idx] + c * meanWL(F1_List, Delta_1)
    if len(F2_List) == 0:
        pass
    else:
        muF2[F2_idx] = (1 - c) * muF2[F2_idx] + c * meanWL(F2_List, Delta_2)

    if len(Cr1_List) == 0:
        pass
    else:
        muCr1[Cr1_idx] = (1 - c) * muCr1[Cr1_idx] + c * meanWA(Cr1_List, Delta_1)
    if len(Cr2_List) == 0:
        pass
    else:
        muCr2[Cr2_idx] = (1 - c) * muCr2[Cr2_idx] + c * meanWA(Cr2_List, Delta_2)


def RunPSHADE():
    global curFEs, TrialRuns, Pop, FitPop, DimSize, muF1, muF2, muCr1, muCr2
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        Initialization()
        Best_list.append(min(FitPop))
        while curFEs < MaxFEs:
            PSHADE()
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./PSHADE_Data/CEC2017/" + "F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    LB = [-100] * dim
    UB = [100] * dim

    for i in range(1, 31):
        FuncNum = i
        if i == 2:
            continue
        RunPSHADE()


if __name__ == "__main__":
    if os.path.exists('PSHADE_Data/CEC2017') == False:
        os.makedirs('PSHADE_Data/CEC2017')
    for dim in [10, 30, 50, 100]:
        main(dim)
