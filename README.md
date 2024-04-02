## A Single-Loop Robust Policy Gradient Method for Robust Markov Decision Processes

This repository contains files that describe the RMDPs experiments in paper "A Single-loop Robust Policy Gradient Method for Robust Markov Decision Processes".



All code are modified from [repository](https://github.com/JerrisonWang/ICML-DRPG).



There are two folders. One is for  `Garnet Problem`  and  `Inventory Problem`.

- Garnet Problem:
  - C++ codes for generating Garnet problems with different size
  - SRPG and DRPG implementation
  - If you want to find another benchmark robust value iteration, please refer to [repository](https://github.com/JerrisonWang/ICML-DRPG). Since we consider gradient-based method in our paper.
- Inventory
  - Python code for generating Inventory problem
  - Python codes for generating a inventory problem with parameterized transition and applying DRPG and SRPG to solve it
  - Python codes for generating a inventory problem with parameterized transition and parameterized policy and applying DRPG and SRPG to solve it









Reference:

[1] Wang, Qiuhao, Chin Pang Ho, and Marek Petrik. "Policy gradient in robust MDPs with global convergence guarantee." In *International Conference on Machine Learning*, pp. 35763-35797. PMLR, 2023.
