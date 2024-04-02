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
from scipy.sparse.linalg import cg, cgs
import cProfile
from scipy.sparse import csr_matrix, vstack

# --------------------------------------------------------------------------- #
# Parse command line arguments:
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='inventory_mdp')
# parser.add_argument('--seed', default=1, type=int, help='seed for random number generators')
parser.add_argument('--sNum', default=10, type=int, help='state number')
parser.add_argument('--aNum', default=5, type=int, help='action number')
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
    Create a randomized MDP object
    
    Parameters
    ----------
    sNum                : state number
    aNum                : action number
    gamma               : discount factor
    empirP              : numpy array of shape (S,SA)
    cost                : numpy array of shape (SA,S)
    rho                 : initial distribution numpy array of shape (1,S)
    lambdDim            : dimension of lambd
    thetaDim            : dimension of theta
    thetaKappa          : kappa for theta, see eq(74) of https://proceedings.mlr.press/v202/wang23i.html or our paper
    lambdKappa          : kappa for lambd, see eq(74) of https://proceedings.mlr.press/v202/wang23i.html or our paper
    thetaCenter         : center parameter for theta
    lambdCenter         : center parameter for lambd
    '''
    def __init__(self, sNum, aNum, gamma, rho, lambdDim, thetaDim, wDim, thetaKappa, lambdKappa, thetaCenter, lambdCenter) -> None:
        self.sNum = sNum
        self.aNum = aNum
        self.gamma = gamma
        self.empirP = None
        self.rho = rho
        self.cost = None
        self.lambdDim = lambdDim
        self.thetaDim = thetaDim
        self.thetaKappa = thetaKappa
        self.lambdKappa = lambdKappa
        self.lambdCenter = lambdCenter
        self.thetaCenter = thetaCenter
        self.wDim = wDim
        self.count = 0
        self.countTest = 0
        
        ## auxiliary
        self.eye = np.identity(self.sNum)
        self.x0_cg_warm = np.random.random(size = (self.sNum))
        self.x0_cg_warm /= np.linalg.norm(self.x0_cg_warm)
        self.x0_cg_warm = None
        self.x0_cg_warm_phiVal = np.random.random(size = (self.sNum))
        self.x0_cg_warm_phiVal /= np.linalg.norm(self.x0_cg_warm_phiVal)
        self.x0_cg_warm_phiVal = None
        I_sparse = csr_matrix(np.identity(self.sNum * self.aNum))
        rows_to_repeat = []

        for row_index in range(I_sparse.shape[0]):
            row = I_sparse.getrow(row_index)
            for _ in range(self.sNum):
                rows_to_repeat.append(row)

        self.eye_repeat = vstack(rows_to_repeat)
    
    def set_garnet_sa(self, seed = 1):
        random.seed(seed)
        ## generate nominal transition kernel
        empirP = np.zeros((self.sNum, self.aNum * self.sNum))
        for i in range(self.sNum * self.aNum):
            # generate transition kernel
            p_mid = np.random.rand(self.sNum)
            p_mid /= np.sum(p_mid)
            for sPlus in range(self.sNum):
                empirP[sPlus, i] = p_mid[sPlus]

        # cost function chosed in [0, 10] randomly
        cost = np.zeros((self.aNum * self.sNum, self.sNum))
        for i in range(self.aNum * self.sNum):
            for j in range(self.sNum):
                cost[i, j] = round(random.gauss(2, 2), 2)
        
        self.empirP = empirP
        self.cost = abs(cost)
    
    def occupancy_measure(self, policy, tranKer):
        '''occupancy measure of a policy
        See eq(3) of https://proceedings.mlr.press/v202/wang23i.html
        return a numpy array of shape (1,S)
        '''
        # P = np.zeros((self.sNum, self.sNum))
        # for s in range(self.sNum):
        #     for a in range(self.aNum):
        #         P[s] += policy[s, a] * tranKer[:, self.aNum * s + a]
        tranKer_reshape = tranKer.reshape([self.sNum, self.sNum, self.aNum]).transpose(1, 0, 2)
        policy_reshape = policy.reshape([self.sNum, self.aNum, 1])
        P = (tranKer_reshape @ policy_reshape).reshape([self.sNum, self.sNum])
        # P_1 = np.linalg.inv(self.eye - self.gamma * P)
        # eta = (1 - self.gamma) * np.dot(self.rho, P_1)
        
        ## replace with cg
        if self.x0_cg_warm is None:
            self.x0_cg_warm, _ = cgs(self.eye - self.gamma * P, self.rho.ravel())
        else:
            self.x0_cg_warm, _ = cgs(self.eye - self.gamma * P, self.rho.ravel(), x0 = self.x0_cg_warm)
        # print(self.count, self.x0_cg_warm)
        # self.count += 1
        eta = (1 - self.gamma) * self.x0_cg_warm
        return eta.ravel()
    
    def occupancy_measure_phiVal(self, policy, tranKer):
        '''occupancy measure of a policy
        See eq(3) of https://proceedings.mlr.press/v202/wang23i.html
        return a numpy array of shape (1,S)
        '''
        # P = np.zeros((self.sNum, self.sNum))
        # for s in range(self.sNum):
        #     for a in range(self.aNum):
        #         P[s] += policy[s, a] * tranKer[:, self.aNum * s + a]
        tranKer_reshape = tranKer.reshape([self.sNum, self.sNum, self.aNum]).transpose(1, 0, 2)
        policy_reshape = policy.reshape([self.sNum, self.aNum, 1])
        P = (tranKer_reshape @ policy_reshape).reshape([self.sNum, self.sNum])
        # P_1 = np.linalg.inv(self.eye - self.gamma * P)
        # eta = (1 - self.gamma) * np.dot(self.rho, P_1)
        
        ## replace with cg
        self.x0_cg_warm_phiVal, _ = cgs(self.eye - self.gamma * P, self.rho.ravel(), x0 = self.x0_cg_warm_phiVal)
        # print(self.count, self.x0_cg_warm)
        # self.count += 1
        eta = (1 - self.gamma) * self.x0_cg_warm_phiVal
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
        
        # p_pi = np.zeros((self.sNum, self.sNum))
        # for s in range(self.sNum):
        #     for a in range(self.aNum):
        #         p_pi[s] += policy[s, a] * tranKer[:, self.aNum * s + a]
        tranKer_reshape = tranKer.reshape([self.sNum, self.sNum, self.aNum]).transpose(1, 0, 2)
        policy_reshape = policy.reshape([self.sNum, self.aNum, 1])
        p_pi = (tranKer_reshape @ policy_reshape).reshape([self.sNum, self.sNum])
        mid = np.linalg.inv(I - self.gamma * p_pi)
        self.v = np.dot(mid, c)
        
        # mid = I - self.gamma * p_pi
        # self.v, _ = cg(mid, c, x0 = self.v)
        return self.v

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
                # for sPlus in range(self.sNum):
                #     expMatTemp[sPlus, self.aNum * s + a] = thetaDotPhi[sPlus] / lambdDotPhi[self.aNum * s + a]
                expMatTemp[:, self.aNum * s + a] = thetaDotPhi / lambdDotPhi[self.aNum * s + a]
        
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
            
            This is a high performance implementation of grad_theta_lamb in another file
        '''    
        gamma_valueF = self.gamma * valueF
        cost_gamma_valueF = self.cost + np.tile(gamma_valueF.reshape([1, -1]), (self.sNum * self.aNum, 1))
        eta_policy = np.multiply(eta[:, np.newaxis], policy)
        tranKer_phiThetaMat = self.phiThetaMat @ tranKer
        phiThetaMat_tile = np.tile(self.phiThetaMat, (1, self.sNum * self.aNum))
        tranKer_phiThetaMat_repeat = np.repeat(tranKer_phiThetaMat, self.sNum, axis = 1)
        phiThetaMat_minus_tranKer_phiThetaMat = phiThetaMat_tile - tranKer_phiThetaMat_repeat
        lambdSA_repeat = np.tile(np.repeat(lambdDotPhi.reshape([1, -1]), self.sNum, axis = 1), (len(self.phiThetaMat), 1))
        phiThetaMat_minus_tranKer_phiThetaMat_div_lambdSA = phiThetaMat_minus_tranKer_phiThetaMat / lambdSA_repeat
        
        cost_gamma_valueF_repeat = np.tile(cost_gamma_valueF.reshape([1, -1]), (len(self.phiThetaMat), 1))
        phiThetaMat_minus_tranKer_phiThetaMat_div_lambdSA_cost_gamma_valueF_repeat\
            = np.multiply(phiThetaMat_minus_tranKer_phiThetaMat_div_lambdSA, cost_gamma_valueF_repeat)
        
        eta_policy_repeat = np.tile(eta_policy.reshape([1, -1]), (self.sNum, 1))
        transKer_eta_policy = np.multiply(eta_policy_repeat, tranKer)
        tranKer_flatten = transKer_eta_policy.transpose().reshape([-1, 1])
        grad_theta = phiThetaMat_minus_tranKer_phiThetaMat_div_lambdSA_cost_gamma_valueF_repeat @ tranKer_flatten
        grad_theta = grad_theta.ravel()
        grad_theta *= (1 / (1 - self.gamma))
        # print("grad_theta:", grad_theta, grad_theta.shape)
        
        thetaDotPhi_repeat = (np.tile(thetaDotPhi.reshape([-1, 1]), (self.aNum * self.sNum, 1))).ravel()
        lambdDotPhiSquare_repeat = (np.repeat(lambdDotPhiSquare.reshape([-1, 1]), self.sNum, axis = 0)).ravel()
        thetaDotPhi_div_lambdDotPhiSquare = thetaDotPhi_repeat/lambdDotPhiSquare_repeat
        
        tranKer_thetaDotPhi_div_lambdDotPhiSquare = np.multiply(tranKer.transpose().ravel(), thetaDotPhi_div_lambdDotPhiSquare)
        
        # eye_repeat = np.repeat(np.identity(self.sNum * self.aNum), self.sNum, axis = 0)        
        
        tranKer_thetaDotPhi_div_lambdDotPhiSquare_localSum = (tranKer_thetaDotPhi_div_lambdDotPhiSquare.reshape([1, -1]) @ self.eye_repeat).ravel()
        tranKer_thetaDotPhi_div_lambdDotPhiSquare_localSum_repeat = np.repeat(tranKer_thetaDotPhi_div_lambdDotPhiSquare_localSum.reshape([1, -1]), self.sNum, axis = 1)
        phiLambdMat_repeat = np.repeat(self.phiLambdMat, self.sNum, axis = 1)
        temp1 = tranKer_thetaDotPhi_div_lambdDotPhiSquare_localSum_repeat - thetaDotPhi_div_lambdDotPhiSquare
        temp1_repeat = np.tile(temp1, (len(self.phiLambdMat), 1))
        temp = np.multiply(temp1_repeat, phiLambdMat_repeat)
        
        cost_gamma_valueF_repeat = np.tile(cost_gamma_valueF.reshape([1, -1]), (len(self.phiLambdMat), 1))
        D2 = np.multiply(temp, cost_gamma_valueF_repeat)
        grad_lambd = (D2 @ tranKer_flatten).ravel()
        grad_lambd *= (1 / (1 - self.gamma))
        
        # print("grad_lambd:", grad_lambd, grad_lambd.shape)
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
        '''
        translation from for loop        
        '''
        gamma_valueF_repeat = self.gamma * np.tile(valueF, (self.aNum * self.sNum, 1))
        cost_gamma_valueF = self.cost + gamma_valueF_repeat
        tranKer_tr = tranKer.transpose()
        tranKer_cost_gamma_valueF = np.multiply(tranKer_tr, cost_gamma_valueF)
        qSA = np.sum(tranKer_cost_gamma_valueF, axis = 1)
        eta_repeat = np.repeat(eta.reshape(-1, 1), self.aNum, axis = 0)
        grad_new = np.multiply(qSA.reshape(-1, 1), eta_repeat)
        grad = grad_new.reshape([self.sNum, self.aNum])
        grad *= (1 / (1 - self.gamma))
        return grad
    
    def grad_policy_para(self, tranKer, eta, valueF, w, policy):
        grad_policy = self.grad_policy(tranKer, eta, valueF)
        policy_grad_policy = -1 * np.multiply(grad_policy, policy)
        # grad = np.zeros_like(w)
        gradNew = np.zeros_like(w)
        for s in range(self.sNum):
            temp = np.tile((self.phiLambdMat[:, s * self.aNum: (s + 1) * self.aNum] @ policy[s, :]).reshape([-1, 1]), (1, self.aNum))
            temp -= self.phiLambdMat[:, s * self.aNum: (s + 1) * self.aNum]
            temp = temp @ policy_grad_policy[s, :]
            gradNew += temp
            # for a in range(self.aNum):
            #     # subMat dimension: m x aNum
            #     subMat = self.phiLambdMat[:, s * self.aNum: (s + 1) * self.aNum] \
            #         - self.phiLambdMat[:, s * self.aNum + a][:, np.newaxis]
            #     subMat = subMat.transpose()
                
            #     subMatMul = subMat @ w
            #     subMatMul -= np.max(subMatMul)
            #     subExpMat = np.exp(subMatMul)
            #     denominator = np.sum(subExpMat)
            #     numerator = subMat.transpose() @ subExpMat
            #     grad += (policy_grad_policy[s, a]) * numerator / denominator
        return gradNew
        
    def grad_policy_para_DSGDA(self, tranKer, eta, valueF, w, policy, wBar, r1):
        grad = self.grad_policy_para(tranKer, eta, valueF, w, policy)
        grad += r1 * (w - wBar)
        return grad
    
    def update_var_w(self, old_w, stepSize, grad):
        new_w = old_w - stepSize * grad
        print("grad norm", np.linalg.norm(grad))
        # new_w = new_w / max(np.linalg.norm(new_w), 3)
        return new_w
    
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
        '''
        return dimension is dim x sNum
        '''
        dimensions = len(args)
        phi = np.zeros((dimensions, self.sNum))
        sigma = 2
        for s in range(self.sNum):
            for dim in range(dimensions):
                up = -(np.linalg.norm(state[:, s] - args[dim])) ** 2
                down = 2 * sigma ** 2
                phi[dim, s] = np.exp(up / down)/ (np.sqrt(2*(sigma**2)*math.pi))
        self.phiThetaMat = phi
        self.phiLambdMatTile = np.tile(phi, (1, self.sNum * self.aNum))

    def phiLambd(self, state, action, *args):
        dimensions = len(args)
        phi = np.zeros((dimensions, self.sNum * self.aNum))
        sigma = 4
        for s in range(self.sNum):
            for a in range(self.aNum):
                mid = list(state[:, s])
                mid.append(action[a])
                stateAction = np.array(mid)
                ## two dimension center
                for m in range(dimensions):
                    up = -(np.linalg.norm(stateAction - args[m])) ** 2
                    down = 2 * sigma ** 2
                    phi[m, self.aNum * s + a] = np.exp(up / down) / (np.sqrt(2*(2**2)*math.pi))
        self.phiLambdMat = phi
    
    def w_to_policy(self, w):
        lineVal = self.phiLambdMat.transpose() @ w
        lineVal = lineVal.reshape([self.sNum, self.aNum]).transpose()
        maxLinearVal = np.tile(np.max(lineVal, axis = 0).reshape(1, -1), (self.aNum, 1))
        lineVal -= maxLinearVal
        explineVal = np.exp(lineVal)
        prob = explineVal / np.sum(explineVal, axis = 0)
        
        policy = prob.transpose()
        # for s in range(self.sNum):
        #     subMat = self.phiLambdMat[:, s * self.aNum: (s + 1) * self.aNum]
        #     subMatMul = subMat.transpose() @ w
        #     subMatMulMax = np.max(subMatMul)
        #     subMatMul -= subMatMulMax
        #     factor = np.exp(subMatMul)
        #     policy[s, :] = factor / np.sum(np.exp(subMatMul))
            # for a in range(self.aNum):
            #     subMat = self.phiLambdMat[:, s * self.aNum: (s + 1) * self.aNum]\
            #     - self.phiLambdMat[:, s * self.aNum + a][:, np.newaxis]
                
            #     subMat = subMat.transpose()
            #     subMatMul = subMat @ w
            #     subMatMulMax = np.max(subMatMul)
            #     subMatMul -= subMatMulMax
            #     subExpMat = np.exp(subMatMul)
            #     policy[s, a] = np.exp(-subMatMulMax) / sum(subExpMat)
        # print("policy:", policy)
        return policy
    
    def phiVal(self, policy, theta0, lambd0, stepSizeLambd, stepSizeTheta):
        lambd = lambd0
        theta = theta0
        t = 0
        Vnew = np.zeros(self.sNum)
        while (True):
            t += 1
            tranKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi = self.paraTranKer(theta, lambd)
            Vnow = self.policy_value(policy, tranKer)
            eta = self.occupancy_measure_phiVal(policy, tranKer)
            gradTheta, gradLambd = self.grad_theta_lambd(policy, eta, Vnow, tranKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi)
            theta_new = self.updateTheta(theta, stepSizeTheta, gradTheta)
            lambd_new = self.updateLambd(lambd, stepSizeLambd, gradLambd)
            gap = np.linalg.norm(Vnew - Vnow)
            phi = np.dot(self.rho, Vnow)[0]
            # print("phi calculate", t)
            if t == 200:
                print("warning: phiVal iter reach 200", phi)
                break
            if gap <= 1e-5:
                break
            Vnew = Vnow
            theta = theta_new
            lambd = lambd_new
        print("phi:", phi)
        return phi
    
    def DRPG(self, stepSizeTheta, stepSizeLambd, stepSizeW, outerLoopIter, innerLoopIter, maxIterAll, logFile = None):
        '''DRPG algorithm
        '''
        w = np.random.random(size=self.wDim)
        w /= np.linalg.norm(w) * 10
        
        lambd = np.random.random(self.lambdDim)
        lambd /= np.linalg.norm(lambd)
        theta = np.random.random(self.thetaDim)
        theta /= np.linalg.norm(theta)
        
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
                if gap <= 1e-5:
                    break
                Vnew = Vnow
                theta = theta_new
                lambd = lambd_new
            return theta, lambd, J_history_inner, phiHis_inner
        J_history = []
        phiHis = []
        tranKer, _, _, _ = self.paraTranKer(theta, lambd)
        policy = self.w_to_policy(w)
        Vnow = self.policy_value(policy, tranKer)
        J = np.dot(self.rho, Vnow)[0]
        # phi = self.phiVal(policy, theta, lambd, 0.01, 0.01)
        maxIterAllcopy = maxIterAll
        # outer loop
        for t in range(outerLoopIter):
            policy = self.w_to_policy(w)
            theta, lambd, J_history_inner, phiHis_inner = innerLoop(policy, theta, lambd, stepSizeLambd, stepSizeTheta, J)
            tranKer, _, _, _ = self.paraTranKer(theta, lambd)
            Vnow = self.policy_value(policy, tranKer)
            eta = self.occupancy_measure(policy, tranKer)
            gradw = self.grad_policy_para(tranKer, eta, Vnow, w, policy)
            w = self.update_var_w(w, stepSizeW, gradw)
            J_history += J_history_inner
            phiHis += phiHis_inner
            J = np.dot(self.rho, Vnow)[0]
            J_history.append(J)
            # phi = self.phiVal(policy,
            #                     theta, lambd,
            #                     0.01,
            #                     0.01)
            phiHis.append(J)
            maxIterAll -= (len(J_history_inner) + 1)
            if maxIterAll <= 0:
                break
            print("inner loop iter: {}, J: {}, phiHis: {}".format(len(J_history_inner), J_history_inner[-1], phiHis_inner[-1]))
        if logFile is not None:
            for i in range(maxIterAllcopy):
                print("{},{},{}".format(i, J_history[i],phiHis[i]), file=logFile)
        return policy, theta, lambd, J_history

    def DSGDA(self, stepSizeTheta, stepSizeLambd, stepSizeW, r1, r2, beta, mu, maxIter, logFile = None):
        # initialize theta, lambd, w
        w = np.random.random(self.wDim)
        w /= np.linalg.norm(w) * 10
        lambd = np.random.random(self.lambdDim)
        lambd /= np.linalg.norm(lambd)
        theta = np.random.random(self.thetaDim)
        theta /= np.linalg.norm(theta)
        wBar = w.copy()
        thetaBar = theta.copy()
        lambdBar = lambd.copy()
        J_history = []
        phiHis = []
        transKer, lambdDotPhi, lambdDotPhiSquare, thetaDotPhi = self.paraTranKer(theta, lambd)
        policy = self.w_to_policy(w)
        eta = self.occupancy_measure(policy, transKer)
        Vnow = self.policy_value(policy, transKer)
        
        for _ in range(maxIter):
            # update policy
            gradw = self.grad_policy_para_DSGDA(transKer, eta, Vnow, w, policy, wBar, r1)
            w = self.update_var_w(w, stepSizeW, gradw)
            policy = self.w_to_policy(w)
            
            #update theta and lambd
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
            wBar = wBar + beta * (w - wBar)
            thetaBar = thetaBar + mu * (theta - thetaBar)
            lambdBar = lambdBar + mu * (lambd - lambdBar)
            J_history.append(np.dot(self.rho, Vnow)[0])
            
            # phi = self.phiVal(policy, theta, lambd, 0.01, 0.01)
            # print("iter: {}, J: {}, phi:{}".format(len(J_history), J_history[-1], phiHis[-1]))
            print("iter: {}, J: {}".format(len(J_history), J_history[-1]))
        if logFile is not None:
            for i in range(len(J_history)):
                print("{},{}".format(i, J_history[i]), file=logFile)
        
        return w, theta, lambd, J_history
    

def inventory_simulation(seed = 1):
    args = parser.parse_args()
    sNum = args.sNum
    aNum = args.aNum
    gamma = 0.95
    lambdDim = 10
    thetaDim = 9
    wDim = lambdDim
    np.random.seed(seed = seed)
    state = np.random.random(size = (2, sNum))
    state[0, :] = np.linspace(1, 5, sNum, endpoint=False)
    state[1, :] = np.random.uniform(-10, 10, sNum)
    action = np.linspace(-10, 10, aNum, endpoint=False)
    thetaKappa = 0.25
    lambdKappa = 0.25
    lambdCenter = np.random.random(size = (lambdDim, ))
    thetaCenter = np.random.random(size = (thetaDim, ))
    rho = randRho(sNum)
    mdp = garnet_mdp(sNum, aNum, gamma, rho, lambdDim, thetaDim, wDim, thetaKappa, lambdKappa, thetaCenter, lambdCenter)
    
    # feature function of phi
    c_s = []
    c_sa = []
    for _ in range(wDim):
        c_sa.append(np.random.random(size = (3, )) * 2 - 4)
    for _ in range(thetaDim):
        c_s.append(np.random.random(size = (2, )) * 2 - 4)
            
    mdp.phiTheta(state, *c_s)
    mdp.phiLambd(state, action, *c_sa)
    
    mdp.set_garnet_sa(seed = seed)
    stepSizeTheta = 0.01
    stepSizeLambd = 0.01
    stepSizeW = 0.5
    outerLoop = 100000 # large enough, stop by maxIterAll
    innerLoop = 200
    maxIterAll = 15000
    logfile = './paraRes/drpg_{}_{}_{}.csv'.format(sNum, aNum, stepSizeW)
    
    with open(logfile, 'w') as f:
        print('iter,J,phi', file = f)
        start_time = time.time()
        mdp.DRPG(stepSizeTheta, stepSizeLambd, stepSizeW, outerLoop, innerLoop, maxIterAll, logFile = f)
        end_time = time.time()
        print("elapsed_time:{}".format(end_time - start_time), file = f)
    
    stepSizeTheta = 0.01
    stepSizeLambd = 0.01
    stepSizeW = 0.5
    # stepSizeW = 0.9
    maxIter = int(maxIterAll / 2)
    r1 = 0.5
    r2 = 0.5
    beta = 0.9
    mu = 0.9
    logfile = './paraRes/srpg_{}_{}_{}.csv'.format(sNum, aNum, stepSizeW)
    with open(logfile, 'w') as f:
        print('iter,J', file = f)
        start_time = time.time()
        mdp.DSGDA(stepSizeTheta, stepSizeLambd, stepSizeW, r1, r2, beta, mu, maxIter, logFile = f)
        end_time = time.time()
        print("elapsed_time:{}".format(end_time - start_time), file = f)
    
    
if __name__ == '__main__':
    inventory_simulation()
    # cProfile.run('inventory_simulation()', 'profile_stats')