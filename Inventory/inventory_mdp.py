'''python
'''

import sys
import os
import numpy as np
import random
import math
import cvxpy as cp
import pandas as pd
import time
import argparse

# --------------------------------------------------------------------------- #
# Parse command line arguments:
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='inventory_mdp')
parser.add_argument('--seed', default=1, type=int, help='seed for random number generators')
# --------------------------------------------------------------------------- #


def randRho(sNum):
    '''Generate a random initial distribution
    '''
    rho = np.random.rand(1, sNum)
    rho /= np.sum(rho)
    return rho

def randPolicy(sNum, aNum):
    '''Generate a random policy
    '''
    policy = np.zeros((sNum, aNum))
    for s in range(sNum):
        policy[s] = np.random.rand(aNum)
        policy[s] /= np.sum(policy[s])
    return policy


class garnet_mdp():
    '''
    Creates a randomized MDP object to pass to an RL algorithm.
    
    Parameters
    ----------
    sNum               : state number
    aNum               : action number
    gamma               : discount factor
    empirP              : numpy array of shape (S,SA)
    branch_matrix       : numpy array of shape (S,SA), 1 means the branch is inactive, 0 means active
    cost                : numpy array of shape (SA,S)
    rho                 : initial distribution numpy array of shape (1,S)
    lambdDim            : dimension of lambd
    thetaDim            : dimension of theta
    thetaKappa          : kappa for theta, see eq(74) of https://proceedings.mlr.press/v202/wang23i.html or our paper
    lambdKappa          : kappa for lambd, see eq(74) of https://proceedings.mlr.press/v202/wang23i.html or our paper
    thetaCenter         : center parameter for theta
    lambdCenter         : center parameter for lambd
    '''
    def __init__(self, sNum, aNum, gamma, rho, lambdDim, thetaDim, thetaKappa, lambdKappa, thetaCenter, lambdCenter) -> None:
        self.sNum = sNum
        self.aNum = aNum
        self.gamma = gamma
        self.empirP = None
        self.branch_matrix = None
        self.rho = rho
        self.cost = None
        self.lambdDim = lambdDim
        self.thetaDim = thetaDim
        self.thetaKappa = thetaKappa
        self.lambdKappa = lambdKappa
        self.lambdCenter = lambdCenter
        self.thetaCenter = thetaCenter
        
    def set_garnet_sa(self, branch_num, seed = 1):
        random.seed(seed)
        # generate nominal transition kernel
        branch_matrix = np.ones((self.sNum, self.aNum * self.sNum))
        empirP = np.zeros((self.sNum, self.aNum * self.sNum))
        for i in range(self.sNum * self.aNum):
            # generate transition kernel
            p_mid = np.random.rand(branch_num)
            p_mid /= np.sum(p_mid)
            ary_idx = list(range(self.sNum))
            idx = random.sample(ary_idx, branch_num)
            for sPlus in range(branch_num):
                empirP[idx[sPlus], i] = p_mid[sPlus]
                branch_matrix[idx[sPlus], i] = 0
        # cost function chosed in [0, 10] randomly
        cost = np.zeros((self.aNum * self.sNum, self.sNum))
        for i in range(self.aNum * self.sNum):
            for j in range(self.sNum):
                cost[i, j] = round(random.gauss(2, 2), 2)
        self.empirP = empirP
        self.branch_matrix = branch_matrix
        self.cost = abs(cost)
    
    def statistic(self):
        print("empirP sum at column:")
        if np.sum(self.empirP, axis=0).all() == 1:
            print("True")
        
        print("sNum: ", self.sNum)
        print("aNum: ", self.aNum)
        print("gamma: ", self.gamma)
        print("rho: ", self.rho)
        print("empirP shape: ", self.empirP.shape)
        print("branch_matrix shape: ", self.branch_matrix.shape)
        print("cost shape: ", self.cost.shape)
        
    
    def occupancy_measure(self, policy, tranKer):
        '''occupancy measure of a policy
        See eq(3) of https://proceedings.mlr.press/v202/wang23i.html
        return a numpy array of shape (1,S)
        '''
        P = np.zeros((self.sNum, self.sNum))
        for s in range(self.sNum):
            for a in range(self.aNum):
                P[s] += policy[s, a] * tranKer[:, self.aNum * s + a]
        eye = np.identity(self.sNum)
        P_1 = np.linalg.inv(eye - self.gamma * P)
        eta = (1 - self.gamma) * np.dot(self.rho, P_1)
        return eta.ravel()
    
    def policy_value(self, policy, tranKer):
        '''Compuate the value of a policy
        ----------------
        tranKer: transition kernel
        '''
        I = np.identity(self.sNum)
        c = np.zeros(self.sNum)
        for s in range(self.sNum):
            for a in range(self.aNum):
                c[s] += policy[s, a] * np.dot(tranKer[:, self.aNum * s + a], self.cost[self.aNum * s + a])
        p_pi = np.zeros((self.sNum, self.sNum))
        for s in range(self.sNum):
            for a in range(self.aNum):
                p_pi[s] += policy[s, a] * tranKer[:, self.aNum * s + a]
        mid = np.linalg.inv(I - self.gamma * p_pi)
        v = np.dot(mid, c)
        return v
    
    def paraTranKer(self, theta, lambd):
        '''Compute the parameterized transition kernel
        ----------------
        Parameters:
            expriP: empirical transition kernel
                shape: (S, SA)
        '''
        # calculate the parameterize transition kernel
        lambdDotPhi = np.zeros(self.sNum * self.aNum)
        lambdDotPhiSquare = np.zeros(self.sNum * self.aNum)
        thetaDotPhi = np.zeros(self.sNum)
        expMat = np.zeros((self.sNum * self.aNum))
        paraTranMat = np.zeros((self.sNum, self.aNum * self.sNum))
        for s in range(self.sNum):
            # prepare for p_sas'^{\xi}
            thetaDotPhi[s] = np.dot(self.phiThetaMat[:, s], theta)
        expMatTemp = np.zeros((self.sNum, self.sNum * self.aNum))
        for s in range(self.sNum):
            for a in range(self.aNum):
                lambdDotPhi[self.aNum * s + a] = np.dot(self.phiLambdMat[:, self.aNum * s + a], lambd)
                lambdDotPhiSquare[self.aNum * s + a] = lambdDotPhi[self.aNum * s + a] ** 2
                for sPlus in range(self.sNum):
                    expMatTemp[sPlus, self.aNum * s + a] = thetaDotPhi[sPlus] / lambdDotPhi[self.aNum * s + a]
        
        maxTemp = np.max(expMatTemp, axis=0)
        for s in range(self.sNum):
            for a in range(self.aNum):
                # subtract the max to avoid overflow
                expMatTemp[:, self.aNum * s + a] -= maxTemp[self.aNum * s + a]
                
        # calculate the exp
        expMat = np.exp(expMatTemp)
        # element-wise multiplication
        paraTranMat = np.multiply(expMat, self.empirP)
        paraTranMat = paraTranMat / np.sum(paraTranMat, axis=0)
        return paraTranMat, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi
    
    def grad_theta_lambd(self, policy, eta, valueF, tranKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi):
        '''Compute the gradient of theta and lambd
        ----------------
        Parameters:
            expriP: empirical transition kernel
                shape: (S, SA)
        '''
        grad_theta = np.zeros(len(self.phiThetaMat))
        for idxTheta in range(len(self.phiThetaMat)):
            for s in range(self.sNum):
                for a in range(self.aNum):
                    p_sa = tranKer[:, self.aNum * s + a]
                    lambdSA = lambdDotPhi[self.aNum * s + a]
                    d = np.zeros(self.sNum)
                    for sPlus in range(self.sNum):
                        d[sPlus]= self.phiThetaMat[idxTheta, sPlus]/ lambdSA
                    d2 = np.zeros(self.sNum)
                    for sPlus in range(self.sNum):
                        gradXi = d[sPlus]- np.dot(p_sa, d) 
                        d2[sPlus]= gradXi * (self.cost[self.aNum * s + a, sPlus]+ self.gamma * valueF[sPlus])
                    grad_theta[idxTheta] += (1 / (1 - self.gamma)) * eta[s] * policy[s, a] * np.dot(p_sa, d2)
        
        grad_lambd = np.zeros(len(self.phiLambdMat))
        for idxLambd in range(len(self.phiLambdMat)):
            for s in range(self.sNum):
                for a in range(self.aNum):
                    p_sa = tranKer[:, self.aNum * s + a]
                    lambdSA = lambdDotPhi[self.aNum * s + a]
                    d = np.zeros(self.sNum)
                    for sPlus in range(self.sNum):
                        d[sPlus]= thetaDotPhi[sPlus]/ lambdDotPhiSquare[self.aNum * s + a]
                        
                    d2 = np.zeros(self.sNum)
                    for sPlus in range(self.sNum):
                        gradXi = (np.dot(p_sa, d) - d[sPlus]) * self.phiLambdMat[idxLambd, self.aNum * s + a]
                        d2[sPlus]= gradXi * (self.cost[self.aNum * s + a, sPlus]+ self.gamma * valueF[sPlus])
                    grad_lambd[idxLambd] += (1 / (1 - self.gamma)) * eta[s] * policy[s, a] * np.dot(p_sa, d2)
                    
        return grad_theta, grad_lambd
    
    def grad_theta_lambd_DSGDA(self,
                               policy,
                               theta,
                               lambd,
                               thetaBar,
                               lambdBar,
                               eta,
                               valueF,
                               paraTranMat,
                               lambdDotPhi,
                               lambdDotPhiSquare,
                               thetaDotPhi,
                               r2):
        grad_theta, grad_lambd = self.grad_theta_lambd(policy, eta, valueF, paraTranMat, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi)
        grad_theta = grad_theta - r2 * (theta - thetaBar)
        grad_lambd = grad_lambd - r2 * (lambd - lambdBar)
        return grad_theta, grad_lambd
    
    def grad_policy(self, tranKer, eta, valueF):
        grad = np.zeros((self.sNum, self.aNum))
        for s in range(self.sNum):
            for a in range(self.aNum):
                qSA = np.dot(tranKer[:, self.aNum * s +a], (self.cost[self.aNum * s + a] + self.gamma * valueF))
                grad[s, a] = eta[s] * qSA / (1 - self.gamma)
        return grad

    def grad_policy_DSGDA(self, tranKer, eta, valueF, policy, policyBar, r1):
        grad = self.grad_policy(tranKer, eta, valueF)
        grad = grad + r1 * (policy - policyBar)
        return grad
    
    def updatePolicy(self, oldPolicy, stepSize, grad):
        '''
        min_{pi \in \Pi} <pi, grad> + 1 / (2 * stepSize) * ||pi - oldPolicy||_2^2
        '''
        oneA = np.ones(self.aNum)
        zeroA = np.zeros(self.aNum)
        newPolicy = np.zeros((self.sNum, self.aNum))
        for s in range(self.sNum):
            policy_s = cp.Variable(self.aNum)
            cons = []
            cons += [policy_s >= zeroA]
            cons += [policy_s @ oneA == 1.0]
            prob = cp.Problem(cp.Minimize(grad[s] @ policy_s + 1 / (2 * stepSize) * cp.sum_squares(policy_s - oldPolicy[s])), cons)
            prob.solve(solver = cp.GUROBI)
            newPolicy[s] = policy_s.value
        return newPolicy
    
    def updateTheta(self, oldTheta, stepSize, gradTheta):
        '''
        min_{theta \in R^d} -<theta, grad> + 1 / (2 * stepSize) * ||theta - oldTheta||_2^2
        '''
        newTheta = cp.Variable(len(oldTheta))
        y = cp.Variable(len(oldTheta))
        oneTheta = np.ones(len(oldTheta))
        cons = []
        # y >= abs(theta - thetaCenter)
        cons += [newTheta - self.thetaCenter <= y]
        cons += [self.thetaCenter - newTheta <= y]
        cons += [oneTheta @ y <= self.thetaKappa]
        # no need additional constraint for branch location
        prob = cp.Problem(cp.Minimize(-1 * gradTheta @ newTheta + 1 / (2 * stepSize) * cp.sum_squares(newTheta - oldTheta)), cons)
        prob.solve(solver = cp.GUROBI)
        return newTheta.value
    
    def updateLambd(self, oldLambd, stepSize, gradLambd):
        '''
        min_{lambd \in R^d} <lambd, grad> + 1 / (2 * stepSize) * ||lambd - oldLambd||_2^2
        '''
        newLambd = cp.Variable(len(oldLambd))
        y = cp.Variable(len(oldLambd))
        oneLambd = np.ones(len(oldLambd))
        zeroLambd = np.zeros(len(oldLambd))
        cons = []
        cons += [newLambd >= 1e-3 * oneLambd]
        # y >= abs(lambd - lambdCenter)
        cons += [newLambd - self.lambdCenter <= y]
        cons += [self.lambdCenter - newLambd <= y]
        cons += [oneLambd @ y <= self.lambdKappa]
        # no need additional constraint for branch location
        prob = cp.Problem(cp.Minimize(-1 * gradLambd @ newLambd + 1 / (2 * stepSize) * cp.sum_squares(newLambd - oldLambd)), cons)
        prob.solve(solver = cp.GUROBI)
        return newLambd.value
                    
    def phiTheta(self, state, *args):
        dimensions = len(args)
        phi = np.zeros((dimensions, self.sNum))
        sigma = 1
        for s in range(self.sNum):
            for dim in range(dimensions):
                up = -(np.linalg.norm(state[:, s] - args[dim])) ** 2
                down = 2 * sigma ** 2
                phi[dim, s] = np.exp(up / down)/ (np.sqrt(2*(sigma**2)*math.pi))
        self.phiThetaMat = phi
    
    def phiLambd(self, state, action, *args):
        dimensions = len(args)
        phi = np.zeros((dimensions, self.sNum * self.aNum))
        for s in range(self.sNum):
            for a in range(self.aNum):
                mid = list(state[:, s])
                mid.append(action[a])
                stateAction = np.array(mid)
                ## two dimension center
                for m in range(dimensions):
                    up = -(np.linalg.norm(stateAction - args[m])) ** 2
                    down = 2 * 2 ** 2
                    phi[m, self.aNum * s + a] = np.exp(up / down) / (np.sqrt(2*(2**2)*math.pi))
        self.phiLambdMat = phi
        
    def phiVal(self, policy, theta0, lambd0, stepSizeLambd, stepSizeTheta):
        lambd = lambd0
        theta = theta0
        t = 0
        Vnew = np.zeros(self.sNum)
        while (True):
            t += 1
            tranKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi = self.paraTranKer(theta, lambd)
            Vnow = self.policy_value(policy, tranKer)
            eta = self.occupancy_measure(policy, tranKer)
            gradTheta, gradLambd = self.grad_theta_lambd(policy, eta, Vnow, tranKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi)
            theta_new = self.updateTheta(theta, stepSizeTheta, gradTheta)
            lambd_new = self.updateLambd(lambd, stepSizeLambd, gradLambd)
            gap = np.linalg.norm(Vnew - Vnow)
            phi = np.dot(self.rho, Vnow)[0]
            if t == 10000:
                print("warning: phiVal iter reach 1000", phi)
                break
            if gap <= 1e-5:
                break
            Vnew = Vnow
            theta = theta_new
            lambd = lambd_new
        return phi
    
    def DRPG(self, stepSizeTheta, stepSizeLambd, stepSizePolicy, outerLoopIter, innerLoopIter, maxIterAll, logFile = None):
        '''DRPG algorithm
        '''
        # initialize theta, lambd, policy
        policy = randPolicy(self.sNum, self.aNum)
        lambd = np.random.random(self.lambdDim)
        theta = np.random.random(self.thetaDim)
        def innerLoop(policy, theta0, lambd0, stepSizeLambd, stepSizeTheta, Jouter):
            lambd = lambd0
            theta = theta0
            t = 0
            Vnew = np.zeros(self.sNum)
            J_history_inner = []
            phiHis_inner = []
            while (t < innerLoopIter):
                t += 1
                tranKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi = self.paraTranKer(theta, lambd)
                Vnow = self.policy_value(policy, tranKer)
                eta = self.occupancy_measure(policy, tranKer)
                gradTheta, gradLambd = self.grad_theta_lambd(policy, eta, Vnow, tranKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi)
                theta_new = self.updateTheta(theta, stepSizeTheta, gradTheta)
                lambd_new = self.updateLambd(lambd, stepSizeLambd, gradLambd)
                gap = np.linalg.norm(Vnew - Vnow)
                J = np.dot(self.rho, Vnow)[0]
                J_history_inner.append(J)
                phiHis_inner.append(Jouter)
                if gap <= 1e-4:
                    break
                Vnew = Vnow
                theta = theta_new
                lambd = lambd_new
            return theta, lambd, J_history_inner, phiHis_inner
        J_history = []
        phiHis = []
        tranKer, _, _, _ = self.paraTranKer(theta, lambd)
        Vnow = self.policy_value(policy, tranKer)
        J = np.dot(self.rho, Vnow)[0]
        maxIterAllcopy = maxIterAll
        phi = self.phiVal(policy,
                            theta, lambd,
                            0.01,
                            0.01)
        for t in range(outerLoopIter):
            theta, lambd, J_history_inner, phiHis_inner = innerLoop(policy, theta, lambd, stepSizeLambd, stepSizeTheta, phi)
            tranKer, _, _, _ = self.paraTranKer(theta, lambd)
            Vnow = self.policy_value(policy, tranKer)
            eta = self.occupancy_measure(policy, tranKer)
            grad_policy = self.grad_policy(tranKer, eta, Vnow)
            policy = self.updatePolicy(policy, stepSizePolicy, grad_policy)
            J_history += J_history_inner
            phiHis += phiHis_inner
            J = np.dot(self.rho, Vnow)[0]
            J_history.append(J)
            phi = self.phiVal(policy,
                                theta, lambd,
                                0.01,
                                0.01)
            phiHis.append(phi)
            maxIterAll -= (len(J_history_inner) + 1)
            if maxIterAll <= 0:
                break
            print("inner loop iter: {}, J: {}, phiHis: {}".format(len(J_history_inner), J_history_inner[-1], phiHis_inner[-1]))
        if logFile is not None:
            for i in range(maxIterAllcopy):
                print("{},{},{}".format(i, J_history[i],phiHis[i]), file=logFile)
        return policy, theta, lambd, J_history
    
    def DSGDA(self, stepSizeTheta, stepSizeLambd, stepSizePolicy, r1, r2, beta, mu, maxIter, logFile = None):
        # initialize theta, lambd, policy
        policy = randPolicy(self.sNum, self.aNum)
        lambd = np.random.random(self.lambdDim)
        theta = np.random.random(self.thetaDim)
        policyBar = policy.copy()
        thetaBar = theta.copy()
        lambdBar = lambd.copy()
        J_history = []
        phiHis = []
        transKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi = self.paraTranKer(theta, lambd)
        eta = self.occupancy_measure(policy, transKer)
        Vnow = self.policy_value(policy, transKer)
        for _ in range(maxIter):
            # update policy
            gradPolicy = self.grad_policy_DSGDA(transKer, eta, Vnow, policy, policyBar, r1)
            policy = self.updatePolicy(policy, stepSizePolicy, gradPolicy)
            # update theta and lambd
            Vnow = self.policy_value(policy, transKer)
            eta = self.occupancy_measure(policy, transKer)
            gradTheta, gradLambd = self.grad_theta_lambd_DSGDA(policy,
                                                               theta,
                                                               lambd,
                                                               thetaBar,
                                                               lambdBar,
                                                               eta,
                                                               Vnow,
                                                               transKer,
                                                               lambdDotPhi,
                                                               lambdDotPhiSquare,
                                                               thetaDotPhi,
                                                               r2)
            theta = self.updateTheta(theta, stepSizeTheta, gradTheta)
            lambd = self.updateLambd(lambd, stepSizeLambd, gradLambd)
            transKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi = self.paraTranKer(theta, lambd)
            # print("theta: {}, lambd: {}".format(np.linalg.norm(theta), np.linalg.norm(lambd)))
            # update piBar
            policyBar = policyBar + beta * (policy - policyBar)
            # update thetaBar and lambdBar
            thetaBar = thetaBar + mu * (theta - thetaBar)
            lambdBar = lambdBar + mu * (lambd - lambdBar)
            J_history.append(np.dot(self.rho, Vnow)[0])
            phi = self.phiVal(policy,
                              theta, lambd,
                              0.01,
                              0.01)
            phiHis.append(phi)
            print("iter: {}, J: {}, phi:{}".format(len(J_history), J_history[-1], phiHis[-1]))
        if logFile is not None:
            for i in range(len(J_history)):
                print("{},{},{}".format(i, J_history[i],phiHis[i]), file=logFile)
        return policy, theta, lambd, J_history


def inventory_simulation():
    sNum = 8
    aNum = 3
    branch_num = 5
    gamma = 0.95
    lambdDim = 2
    thetaDim = 2
    state = np.array([[0.25, 0.5, 0.75, 1, 0.25, 0.5, 0.75, 1], [1.3, -2.1, 3.4, -1, 2.5, 0.5, 1.8, -0.8]])
    action = np.array([-3, -1, 5])
    thetaKappa = 1
    lambdKappa = 1
    lambdCenter = np.array([0.7, 0.6])
    thetaCenter = np.array([0.4, 0.9])
    args = parser.parse_args()
    np.random.seed(1)
    rho = randRho(sNum)
    mdp = garnet_mdp(sNum, aNum, gamma, rho, lambdDim, thetaDim, thetaKappa, lambdKappa, thetaCenter, lambdCenter)
    # feature function of phi
    # every state is 2 dim
    c_s1 = np.array([-1, 2])
    c_s2 = np.array([0.3, -0.6])
    # every action is 1 dim hence c_sa is 3 dim
    c_sa1 = np.array([1.3, 2.1, 1])
    c_sa2 = np.array([-0.7, 1.5, 0.5])
    mdp.phiTheta(state, c_s1, c_s2)
    mdp.phiLambd(state, action, c_sa1, c_sa2)
    mdp.set_garnet_sa(branch_num, seed=1)
    stepSizeTheta = 0.01
    stepSizeLambd = 0.01
    stepSizePolicy = 0.1
    outerLoop = 100000 # large enough, stop by maxIterAll
    innerLoop = 200
    
    seed = args.seed # for initial policy and theta, lambd
    np.random.seed(seed)
    logfile = './drpg{}.csv'.format(seed)
    maxIterAll = 2000
    with open(logfile, 'w') as f:
        print('iter,J,phi', file=f)
        mdp.DRPG(stepSizeTheta, stepSizeLambd, stepSizePolicy, outerLoop, innerLoop, maxIterAll, logFile = f)
    stepSizeTheta = 0.05
    stepSizeLambd = 0.01
    stepSizePolicy = 0.05
    np.random.seed(seed)
    logfile = './dsgda{}.csv'.format(seed)
    maxIter = int(maxIterAll / 2)
    r1 = 15.0
    r2 = 15.0
    beta = 0.3
    mu = 0.3
    with open(logfile, 'w') as f:
        print('iter,J,phi', file=f)
        mdp.DSGDA(stepSizeTheta, stepSizeLambd, stepSizePolicy, r1, r2, beta, mu, maxIter, logFile = f)
                
            
                
if __name__ == '__main__':
    inventory_simulation()