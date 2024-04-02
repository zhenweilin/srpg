#include "garnet.h"
#include <random>
#include <fstream>
#include <functional>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <assert.h>
#include <chrono>
#include <limits>
#include <cmath>
#include <memory>
#include <stdio.h>
#include "gurobi_c++.h"
#include <sstream>
#include <thread>
#include "Eigen/Dense"
using namespace Eigen;

MDP garnet_mdps_sa(size_t nStates, size_t nActions, size_t nBranchs, int seed){
    /*
    generate random MDPs Instance
    */
    // Create random number generator with default seed
    default_random_engine generator;
    generator.seed(seed);
    uniform_real_distribution<double> distribution(0.0, 5.0);
    double gamma = 0.95;

    // randomized initial distribution
    vector<double> rho;
    // allocate memory
    rho.reserve(nStates);
    double rhoSum = 0.0;
    double rhoTemp;
    for (size_t s = 0; s < nStates; ++s) {
        rhoTemp = distribution(generator);
        rhoSum += rhoTemp;
        // add to vector tailor
        rho.push_back(rhoTemp);
    }
    // normalize
    for (size_t s = 0; s < nStates; ++s) {
        rho[s] /= rhoSum;
    }

    // cost function
    vector<vector<vector<double>>> cost; cost.reserve(nStates);
    for (size_t s = 0; s < nStates; ++s) {
        vector<vector<double>> cost_s; cost_s.reserve(nActions);
        for (size_t a = 0; a < nActions; ++a) {
            vector<double> cost_sa; cost_sa.reserve(nStates);
            for (size_t sPlus = 0; sPlus < nStates; ++sPlus) {
                cost_sa.push_back(distribution(generator));
            }
            cost_s.push_back(cost_sa);
        }
        cost.push_back(cost_s);
    }

    // nominal transition and branch location kernel
    vector<vector<vector<double>>> P; P.reserve(nStates);
    vector<vector<vector<double>>> B; B.reserve(nStates);

    for (size_t s = 0; s < nStates; ++s){
        vector<vector<double>> P_s; P_s.reserve(nActions);
        vector<vector<double>> B_s; B_s.reserve(nActions);
        for (size_t a = 0; a < nActions; ++a){
            vector<double> P_sa; P_sa.reserve(nStates); P_sa.assign(nStates, 0.0);
            vector<double> B_sa; B_sa.reserve(nStates); B_sa.assign(nStates, 1.0);
            double P_saSum = 0.0;
            // random choose nBranchs size state from nStates
            vector<int> index = P_branch(nStates, nBranchs, seed);
            for (size_t b = 0; b < nBranchs; b++){
                // randomize the transition kernel
                // since the sum of P_sa at B_sa is zero in constraint
                // which implies that B_sa == 1 is inactive
                // and B_sa == 0 is active
                B_sa[index[b]] = 0.0;
                P_sa[index[b]] = distribution(generator);
                P_saSum += P_sa[index[b]];
            }
            for (size_t sPlus = 0; sPlus < nStates; ++sPlus){
                P_sa[sPlus] /= P_saSum;
            }
            P_s.push_back(P_sa);
            B_s.push_back(B_sa);
        }
        P.push_back(P_s);
        B.push_back(B_s);
    }
    MDP mdpInstance;
    mdpInstance.P = P;
    mdpInstance.B = B;
    mdpInstance.cost = cost;
    mdpInstance.gamma = gamma;
    mdpInstance.rho = rho;
    mdpInstance.createAuxiliaryVar();
    return mdpInstance;
}

vector<vector<double>> random_tolerance_kappa_sa(const MDP& mdpInstance, int seed){
    /*
    generate random tolerance kappa
    */
    vector<vector<double>> kappa; kappa.reserve(mdpInstance.nStates);
    default_random_engine generator;
    generator.seed(seed);
    uniform_real_distribution<double> distribution(0.1, 0.5);
    for (size_t s = 0; s < mdpInstance.nStates; ++s){
        vector<double> kappa_s; kappa_s.reserve(mdpInstance.nActions);
        for (size_t a = 0; a < mdpInstance.nActions; ++a){
            kappa_s.push_back(distribution(generator));
        }
        kappa.push_back(kappa_s);
    }
    return kappa;
}

/***************************************************************
Robust `Value Iteration` for SA-rectangularity RMDP (pi, v_opt)
**************************************************************/

tuple<vector<vector<double>>, vector<vector<vector<double>>>, vector<double>> contraction_mapping(
    const MDP& mdpInstance, const vector<vector<double>>& kappa,
    const vector<double> value0
){
    constexpr double inf = numeric_limits<double>::infinity();
    const vector<vector<vector<double>>>& P = mdpInstance.P;
    const vector<vector<vector<double>>>& B = mdpInstance.B;
    const vector<vector<vector<double>>>& cost = mdpInstance.cost;
    const vector<double>& rho = mdpInstance.rho;
    const double& gamma = mdpInstance.gamma;
    const int& nStates = (int)mdpInstance.nStates;
    const int& nActions = (int)mdpInstance.nActions;
    // set updated value function
    vector<double> valueFinal(nStates, 0.0);
    // initialize action_choice as 0
    vector<vector<double>> action_choice; action_choice.reserve(nStates);
    for (size_t s = 0; s < nStates; ++s){
        vector<double> action_choice_s(nActions, 0.0);
        action_choice.push_back(action_choice_s);
    }

    vector<vector<vector<double>>> PFinal; PFinal.reserve(nStates);
    for (size_t s = 0; s < nStates; ++s){
        vector<vector<double>> PFinal_s; PFinal_s.reserve(nActions);
        for (size_t a = 0; a < nActions; ++a){
            vector<double> PFinal_sa(nStates, 0.0);
            PFinal_s.push_back(PFinal_sa);
        }
        PFinal.push_back(PFinal_s);
    }

    for (int s = 0; s < nStates; ++s){
        int temp_Amin = 0;
        double temp_min = 0.0;
        for (int a = 0; a < nActions; ++a){
            vector<double> temp_c; temp_c.reserve(nStates);
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                temp_c.push_back(cost[s][a][sPlus] + gamma * value0[sPlus]);
            }
            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);
            env.start();
            GRBModel model = GRBModel(env);

            // Create variables, p_sa, y
            GRBVar* p_sa;
            p_sa = model.addVars(nStates, GRB_CONTINUOUS);
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                p_sa[sPlus].set(GRB_DoubleAttr_LB, 0.0);
                p_sa[sPlus].set(GRB_DoubleAttr_UB, 1.0);
            }
            GRBVar* temp_y;
            temp_y = model.addVars(nStates, GRB_CONTINUOUS);
            for (int sPlus = 0; sPlus < nStates; ++sPlus)
            {
                temp_y[sPlus].set(GRB_DoubleAttr_LB, -inf);
                temp_y[sPlus].set(GRB_DoubleAttr_UB, inf);
            }

            // Set objective
            GRBLinExpr obj = 0.0;
            // constraints
            GRBLinExpr sum_p;
            GRBLinExpr sum_y;
            GRBLinExpr sum_bp;
            // p_sa^T 1 = 1
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_p += p_sa[sPlus];
            }
            model.addConstr(sum_p == 1.0);
            // y^T 1 <= kappa_sa
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_y += temp_y[sPlus];
            }
            model.addConstr(sum_y <= kappa[s][a]);

            // y >= abs(p - p_c)
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                model.addConstr(temp_y[sPlus] >= p_sa[sPlus] - P[s][a][sPlus]);
                model.addConstr(temp_y[sPlus] >= P[s][a][sPlus] - p_sa[sPlus]);
            }

            // branch location constraint
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_bp += B[s][a][sPlus] * p_sa[sPlus];
            }
            model.addConstr(sum_bp == 0.0);

            // set objective
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                obj += p_sa[sPlus] * temp_c[sPlus];
            }
            model.setObjective(obj, GRB_MAXIMIZE);

            // Optimize model
            model.optimize();
            double innerP = 0.0;
            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL){
                innerP = model.get(GRB_DoubleAttr_ObjVal);
                temp_Amin = a;
            }
            else{
                cout << "Error: optimization failed" << endl;
                exit(1);
            }
            // PFinal[s][a] = P[s][a];
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                PFinal[s][a][sPlus] = p_sa[sPlus].get(GRB_DoubleAttr_X);
            }
            if (a == 0){
                temp_min = innerP;
            }else{
                if (innerP < temp_min){
                    temp_min = innerP;
                    temp_Amin = a;
                }
            }
        }
        action_choice[s][temp_Amin] = 1.0;
        valueFinal[s] = temp_min;
    }
    return {action_choice, PFinal, valueFinal};
}



tuple<vector<vector<double>>, vector<double>, vector<double>> robustValueIteration(
    // Apply Contraction to solve robust optimal value funtion
    const MDP& mdpInstance,
    const vector<vector<double>>& kappa,
    const int& max_iter,
    const vector<double>& v_ini)
{
    const size_t nStates = mdpInstance.P.size();
    const size_t nActions = mdpInstance.P[0].size();


    // set zero vector as the initial v_0
    vector<double> v_0 = v_ini;
    
    /*
     for (int s = 0; s < nStates; ++s){
         cout << "vIni["<< s << "]" << v_0[s] << "\n";
     }
     */

    // set temp vector
    vector<double> v_1(nStates, 0.0);
    int t = 0;
    vector<double> J; J.reserve(max_iter);
    while (true)
    {
        // question: pi_now is only 0 and 1, not probability
        // there exists a deterministic optimal policy for SA rectangularity
        t += 1;
        auto result = contraction_mapping(mdpInstance, kappa, v_0);
        auto pi_now = std::get<0>(result);
        auto P_now = std::get<1>(result);
        auto v_1 = std::get<2>(result);
        double JIter = inner_product(v_1, mdpInstance.rho);
        J.push_back(JIter);
        // double PhiVal = PHIVal(mdpInstance, pi_now, mdpInstance.P, kappa, 0.01, max_iter);
        if (t == max_iter)
        {
            return { pi_now, v_1, J };
            break;
        }
        v_0 = v_1;
    }
}

/***************************************************************
DRPG for SA-rectangularity RMDP (pi, v_opt)
**************************************************************/

void updateP_one(
                 const MDP& mdpInstance,
                 const vector<vector<double>>& policy,
                 vector<vector<vector<double>>>& PNew,
                 const vector<vector<double>>& kappa,
                 vector<vector<vector<double>>> grad_P,
                 const double& stepsize,
                 int s, int a
                 ){
    int nStates = (int)mdpInstance.nStates;
    int nActions = (int)mdpInstance.nActions;
    const vector<vector<vector<double>>>& P = mdpInstance.P;
    const vector<vector<vector<double>>>& B = mdpInstance.B;
    const vector<vector<vector<double>>>& cost = mdpInstance.cost;
    double inf = numeric_limits<double>::infinity();
    GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model = GRBModel(env);
    // Define decision variables
    GRBVar* p_sa;
    p_sa = model.addVars(nStates, GRB_CONTINUOUS);
    for (int sPlus = 0; sPlus < nStates; ++sPlus){
        p_sa[sPlus].set(GRB_DoubleAttr_LB, 0.0);
        p_sa[sPlus].set(GRB_DoubleAttr_UB, 1.0);
    }
    GRBVar* temp_y;
    temp_y = model.addVars(nStates, GRB_CONTINUOUS);
    for (int sPlus = 0; sPlus < nStates; ++sPlus){
        temp_y[sPlus].set(GRB_DoubleAttr_LB, -inf);
        temp_y[sPlus].set(GRB_DoubleAttr_UB, inf);
    }
    // expression
    GRBQuadExpr obj;
    GRBLinExpr sum_p;
    GRBLinExpr sum_y;
    GRBLinExpr sum_bp;
    // p_sa^T 1 = 1
    for (int sPlus = 0; sPlus < nStates; ++sPlus){
        sum_p += p_sa[sPlus];
    }
    model.addConstr(sum_p == 1.0);
    // y^T 1 <= kappa_sa
    for (int sPlus = 0; sPlus < nStates; ++sPlus){
        sum_y += temp_y[sPlus];
    }
    model.addConstr(sum_y <= kappa[s][a]);
    // y >= abs(p - p_c)
    for (int sPlus = 0; sPlus < nStates; ++sPlus){
        model.addConstr(temp_y[sPlus] >= p_sa[sPlus] - P[s][a][sPlus]);
        model.addConstr(temp_y[sPlus] >= P[s][a][sPlus] - p_sa[sPlus]);
    }
    // branch location constraint
    for (int sPlus = 0; sPlus < nStates; ++sPlus){
        sum_bp += B[s][a][sPlus] * p_sa[sPlus];
    }
    model.addConstr(sum_bp == 0.0);
    // set objective: minimize -<grad, p> + 1/2 * stepsize * ||p - PNew||^2
    for (int sPlus = 0; sPlus < nStates; ++sPlus){
        obj += (p_sa[sPlus] - PNew[s][a][sPlus]) * (p_sa[sPlus] - PNew[s][a][sPlus]);
        obj -= 2 * stepsize * grad_P[s][a][sPlus] * p_sa[sPlus];
    }
    model.setObjective(obj, GRB_MINIMIZE);
    // Optimize model
    model.optimize();
    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL){
        for (int sPlus = 0; sPlus < nStates; ++sPlus){
            PNew[s][a][sPlus] = p_sa[sPlus].get(GRB_DoubleAttr_X);
        }
    }
    else{
        cout << "Error: optimization failed" << endl;
        exit(1);
    }
}

vector<vector<vector<double>>> updateP_SA_multi(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    vector<vector<vector<double>>> PNew,
    const vector<vector<double>>& kappa,
    vector<vector<vector<double>>> grad_P,
    const double& stepsize
                                                ){
    int nStates = (int)mdpInstance.nStates;
    int nActions = (int)mdpInstance.nActions;
    std::vector<std::thread> threads;
    for (int s = 0; s < nStates; ++s){
        for (int a = 0; a < nActions; ++a){
            threads.emplace_back(updateP_one, std::ref(mdpInstance), std::ref(policy), std::ref(PNew), std::ref(kappa), std::ref(grad_P), stepsize, s, a);
        }
    }
    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    return {PNew};
}


vector<vector<vector<double>>> updateP_SA(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    vector<vector<vector<double>>> PNew,
    const vector<vector<double>>& kappa,
    vector<vector<vector<double>>> grad_P,
    const double& stepsize
){
    int nStates = (int)mdpInstance.nStates;
    int nActions = (int)mdpInstance.nActions;
    const vector<vector<vector<double>>>& P = mdpInstance.P;
    const vector<vector<vector<double>>>& B = mdpInstance.B;
    const vector<vector<vector<double>>>& cost = mdpInstance.cost;
    double inf = numeric_limits<double>::infinity();
    for (int s = 0; s < nStates; ++s){
        for (int a = 0; a < nActions; ++a){
            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);
            env.start();
            GRBModel model = GRBModel(env);
            // Define decision variables
            GRBVar* p_sa;
            p_sa = model.addVars(nStates, GRB_CONTINUOUS);
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                p_sa[sPlus].set(GRB_DoubleAttr_LB, 0.0);
                p_sa[sPlus].set(GRB_DoubleAttr_UB, 1.0);
            }
            GRBVar* temp_y;
            temp_y = model.addVars(nStates, GRB_CONTINUOUS);
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                temp_y[sPlus].set(GRB_DoubleAttr_LB, -inf);
                temp_y[sPlus].set(GRB_DoubleAttr_UB, inf);
            }
            // expression
            GRBQuadExpr obj;
            GRBLinExpr sum_p;
            GRBLinExpr sum_y;
            GRBLinExpr sum_bp;
            // p_sa^T 1 = 1
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_p += p_sa[sPlus];
            }
            model.addConstr(sum_p == 1.0);
            // y^T 1 <= kappa_sa
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_y += temp_y[sPlus];
            }
            model.addConstr(sum_y <= kappa[s][a]);
            // y >= abs(p - p_c)
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                model.addConstr(temp_y[sPlus] >= p_sa[sPlus] - P[s][a][sPlus]);
                model.addConstr(temp_y[sPlus] >= P[s][a][sPlus] - p_sa[sPlus]);
            }
            // branch location constraint
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_bp += B[s][a][sPlus] * p_sa[sPlus];
            }
            model.addConstr(sum_bp == 0.0);
            // set objective: minimize -<grad, p> + 1/2 * stepsize * ||p - PNew||^2
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                obj += (p_sa[sPlus] - PNew[s][a][sPlus]) * (p_sa[sPlus] - PNew[s][a][sPlus]);
                obj -= 2 * stepsize * grad_P[s][a][sPlus] * p_sa[sPlus];
            }
            model.setObjective(obj, GRB_MINIMIZE);
            // Optimize model
            model.optimize();
            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL){
                for (int sPlus = 0; sPlus < nStates; ++sPlus){
                    PNew[s][a][sPlus] = p_sa[sPlus].get(GRB_DoubleAttr_X);
                }
            }
            else{
                cout << "Error: optimization failed" << endl;
                exit(1);
            }
            // end a
        }
        // end s
    }
    return {PNew};
}

double PHIVal(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P_0,
    const vector<vector<double>>& kappa,
    const double& stepsize,
    const int& max_iter
){
    /* using projected gradient descent for the inner problem */
    /* inner loop of DRPG */
    constexpr double inf = numeric_limits<double>::infinity();
    // empirical nominal probability
    const vector<vector<vector<double>>>& P = mdpInstance.P;
    const vector<vector<vector<double>>>& B = mdpInstance.B;
    const vector<vector<vector<double>>>& cost = mdpInstance.cost;
    const vector<double>& rho = mdpInstance.rho;
    const double& gamma = mdpInstance.gamma;
    const int nStates = (int) mdpInstance.P.size();
    const int nActions = (int) mdpInstance.P[0].size();
    
    vector<double> valueNew;
    vector<double> valueOld(nStates, 0.0);
    int t = 0;
    // create var for store the maximum transition kernel
    vector<vector<vector<double>>> PNew = P_0;

    while (true){
        t += 1;
        valueNew = mdp_value(mdpInstance, policy, PNew);
        vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
        vector<vector<vector<double>>> grad_P = MDP_grad_P(mdpInstance, policy, PNew, eta, valueNew);
        PNew = updateP_SA_multi(mdpInstance, policy, PNew, kappa, grad_P, stepsize);
        double gap = 0.0;
        for (size_t sPlus = 0; sPlus < nStates; ++sPlus){
            gap += (valueNew[sPlus] - valueOld[sPlus]) * (valueNew[sPlus] - valueOld[sPlus]);
        }
        gap = sqrt(gap);
        if (gap < 1e-4 || t == max_iter){
            double PHI = inner_product(valueNew, rho);
            return PHI;
            break;
        }
        valueOld = valueNew;
    }
}


tuple<vector<vector<vector<double>>>, vector<double>, int> drpg_inner_loop_pgd(
    const MDP& mdpInstance, 
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P_0,
    const vector<vector<double>>& kappa,
    const double& epsilon,
    const double& stepsize,
    const int& max_iter,
    vector<double>& JHis,
    vector<double>& PhiHis,
    double& phiValOuter
){
    /* using projected gradient descent for the inner problem */
    /* inner loop of DRPG */
    constexpr double inf = numeric_limits<double>::infinity();
    // empirical nominal probability
    const vector<vector<vector<double>>>& P = mdpInstance.P;
    const vector<vector<vector<double>>>& B = mdpInstance.B;
    const vector<vector<vector<double>>>& cost = mdpInstance.cost;
    const vector<double>& rho = mdpInstance.rho;
    const double& gamma = mdpInstance.gamma;
    const int nStates = (int) mdpInstance.P.size();
    const int nActions = (int) mdpInstance.P[0].size();
    
    vector<double> valueNew;
    vector<double> valueOld(nStates, 0.0);
    int t = 0;
    // create var for store the maximum transition kernel
    vector<vector<vector<double>>> PTemp = P_0;
    vector<vector<vector<double>>> PNew = P_0;
    /*
     PTemp.reserve(nStates);
     for (size_t s = 0; s < nStates; ++s){
         vector<vector<double>> PTemp_s; PTemp_s.reserve(nActions);
         for (size_t a = 0; a < nActions; ++a){
             vector<double> PTemp_sa(nStates, 0.0);
             PTemp_s.push_back(PTemp_sa);
         }
         PTemp.push_back(PTemp_s);
     }
     */
    double JIter;
    while (true){
        t += 1;
        valueNew = mdp_value(mdpInstance, policy, PNew);
        vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
        vector<vector<vector<double>>> grad_P = MDP_grad_P(mdpInstance, policy, PNew, eta, valueNew);
        PNew = updateP_SA_multi(mdpInstance, policy, PNew, kappa, grad_P, stepsize);
        double gap = 0.0;
        for (size_t sPlus = 0; sPlus < nStates; ++sPlus){
            gap += (valueNew[sPlus] - valueOld[sPlus]) * (valueNew[sPlus] - valueOld[sPlus]);
        }
        gap = sqrt(gap);
        JIter = inner_product(valueNew, rho);
        JHis.push_back(JIter);
        PhiHis.push_back(phiValOuter);
        if (gap < epsilon || t == max_iter){
            return {PTemp, valueNew, t};
            break;
        }
        valueOld = valueNew;
    }
}

void update_policy_sa_one(
  const MDP& mdpInstance,
  const vector<vector<double>>& policy,
  const double& stepsize,
  const vector<vector<double>>& grad_policy,
  vector<vector<double>>& policy_new,
  int s
){
    const int& nActions = (int)mdpInstance.nActions;
    GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model = GRBModel(env);
    // Define decision variables
    GRBVar* policy_s;
    policy_s = model.addVars(nActions, GRB_CONTINUOUS);
    for (int a = 0; a < mdpInstance.nActions; ++a){
        policy_s[a].set(GRB_DoubleAttr_LB, 0.0);
        policy_s[a].set(GRB_DoubleAttr_UB, 1.0);
    }
    // expression
    GRBQuadExpr obj;
    // constraints term
    GRBLinExpr sum_policy;
    // policy^T 1 = 1
    for (int a = 0; a < mdpInstance.nActions; ++a){
        sum_policy += policy_s[a];
    }
    model.addConstr(sum_policy == 1.0);
    // set objective: minimize <grad, policy> + 1/2 * stepsize * ||policy - policy_t||^2
    for (int a = 0; a < mdpInstance.nActions; ++a){
        obj += (policy_s[a] - policy[s][a]) * (policy_s[a] - policy[s][a]);
        obj += 2 * stepsize * grad_policy[s][a] * policy_s[a];
    }
    model.setObjective(obj, GRB_MINIMIZE);
    // Optimize model
    model.optimize();
    for (int a = 0; a < mdpInstance.nActions; ++a){
        policy_new[s][a] = policy_s[a].get(GRB_DoubleAttr_X);
    }
}


vector<vector<double>> update_policy_sa_multi
 (
     const MDP& mdpInstance,
     const vector<vector<double>>& policy,
     const double& stepsize,
     const vector<vector<double>>& grad_policy
  ){
    vector<vector<double>> policy_new(mdpInstance.nStates, vector<double>(mdpInstance.nActions, 0.0));
    std::vector<std::thread> threads;
    for (int s = 0; s < mdpInstance.nStates; ++s){
        threads.emplace_back(update_policy_sa_one, std::ref(mdpInstance), std::ref(policy), stepsize, std::ref(grad_policy), std::ref(policy_new), s);
    }
    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    return policy_new;
}

vector<vector<double>> update_policy_sa(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const double& stepsize,
    const vector<vector<double>>& grad_policy
){
    vector<vector<double>> policy_new; policy_new.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        vector<double> policy_new_s(mdpInstance.nActions, 0.0);
        policy_new.push_back(policy_new_s);
    }
    
    const int& nActions = (int)mdpInstance.nActions;

    for (int s = 0; s < mdpInstance.nStates; ++s){
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 0);
        env.start();
        GRBModel model = GRBModel(env);
        // Define decision variables
        GRBVar* policy_s;
        policy_s = model.addVars(nActions, GRB_CONTINUOUS);
        for (int a = 0; a < mdpInstance.nActions; ++a){
            policy_s[a].set(GRB_DoubleAttr_LB, 0.0);
            policy_s[a].set(GRB_DoubleAttr_UB, 1.0);
        }
        // expression
        GRBQuadExpr obj;
        // constraints term
        GRBLinExpr sum_policy;
        // policy^T 1 = 1
        for (int a = 0; a < mdpInstance.nActions; ++a){
            sum_policy += policy_s[a];
        }
        model.addConstr(sum_policy == 1.0);
        // set objective: minimize <grad, policy> + 1/2 * stepsize * ||policy - policy_t||^2
        for (int a = 0; a < mdpInstance.nActions; ++a){
            obj += (policy_s[a] - policy[s][a]) * (policy_s[a] - policy[s][a]);
            obj += 2 * stepsize * grad_policy[s][a] * policy_s[a];
        }
        model.setObjective(obj, GRB_MINIMIZE);
        // Optimize model
        model.optimize();
        for (int a = 0; a < mdpInstance.nActions; ++a){
            policy_new[s][a] = policy_s[a].get(GRB_DoubleAttr_X);
        }
    }
    return policy_new;
}


tuple<vector<vector<double>>, vector<double>, vector<double>, vector<double>, vector<int>> drpg_sa(
    const MDP& mdpInstance,
    const vector<vector<double>>& kappa,
    const double& stepOuter,
    const double& stepInner,
    const int& max_iter,
    const int& maxIterAll,
    const int& policySeed
){
    // random policy
    vector<vector<double>> policy = random_policy(mdpInstance, policySeed);
    vector<vector<vector<double>>> P = mdpInstance.P;
    int t = 0;
    double inner_eps = 1e-3;
    vector<vector<double>> grad_policy;
    vector<double> eta;
    int maxAllIter = maxIterAll;
    vector<double> JHis; JHis.reserve(maxAllIter);
    double JIter;
    vector<double> phiHis; phiHis.reserve(max_iter);
    vector<int> recordBreakPoint; recordBreakPoint.reserve(max_iter);
    int record = 0;
    double PhiValOuter = PHIVal(mdpInstance, policy, P, kappa, stepInner, max_iter);
    while (true)
    {
        t += 1;
        auto result = drpg_inner_loop_pgd(mdpInstance, policy, P, kappa, inner_eps, stepInner, max_iter, JHis, phiHis, PhiValOuter);
        auto P = std::get<0>(result);
        auto v = std::get<1>(result);
        auto innerIter = std::get<2>(result);
        eta = mdp_occupancy(mdpInstance, P, policy);
        grad_policy = mdp_grad_policy(mdpInstance, policy, P, eta, v);
        policy = update_policy_sa_multi(mdpInstance, policy, stepOuter, grad_policy);
        JIter = inner_product(v, mdpInstance.rho);
        PhiValOuter = PHIVal(mdpInstance, policy, P, kappa, stepInner, max_iter);
        if (t % 5 == 0){
            cout << "drpg method " << "t:" << t << " JIter:" << JIter << ", Phi Val:"<< PhiValOuter << "\n";
        }
        JHis.push_back(JIter);
        maxAllIter -= (innerIter + 1);
        record += (innerIter + 1);
        phiHis.push_back(PhiValOuter);
        recordBreakPoint.push_back(record);
        if (t == max_iter || maxAllIter <= 0)
        {
            phiHis.erase(phiHis.begin() + record, phiHis.end());
            recordBreakPoint.erase(recordBreakPoint.begin() + t, recordBreakPoint.end());
            return { policy, v, JHis, phiHis, recordBreakPoint};
        }
    }
}

tuple<vector<vector<double>>, vector<double>, vector <double>, vector <double>> dsgda_sa(
    const MDP& mdpInstance,
    const vector<vector<double>>& kappa,
    const double& stepPolicy,
    const double& stepP,
    const int& max_iter,
    const double& beta,
    const double& mu,
    const double& r1,
    const double& r2,
    const int& policySeed
){
    // random policy
    vector<vector<double>> policy = random_policy(mdpInstance, policySeed);
    vector<vector<vector<double>>> P = mdpInstance.P;
    vector<vector<vector<double>>> PNew = P;
    vector<vector<double>> policyBar = policy;
    vector<vector<vector<double>>> PBar = P;
    int t = 0;
    vector<vector<double>> grad_policy;
    vector<double> JHis; JHis.reserve(max_iter);
    vector<double> phiHis; phiHis.reserve(max_iter);
    double JIter;
    auto valueNew = mdp_value(mdpInstance, policy, PNew);
    vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
    while (true)
    {
        t += 1;
        grad_policy = dsgda_grad_policy(mdpInstance, policy, policyBar, r1, PNew, eta, valueNew);
        policy = update_policy_sa_multi(mdpInstance, policy, stepPolicy, grad_policy);

        valueNew = mdp_value(mdpInstance, policy, PNew);
        vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
        vector<vector<vector<double>>> grad_P = dsgda_grad_P(mdpInstance, policy, PNew, PBar, r2, eta, valueNew);
        PNew = updateP_SA_multi(mdpInstance, policy, PNew, kappa, grad_P, stepP);

        // update policyBar and PBar
        // policyBar = policyBar + beta * (policy - policyBar);
        for (int s = 0; s < mdpInstance.nStates; ++s){
            for (int a = 0; a < mdpInstance.nActions; ++a){
                policyBar[s][a] += beta * (policy[s][a] - policyBar[s][a]);
            }
        }
        // PBar = PBar + mu * (P - PBar);
        for (int s = 0; s < mdpInstance.nStates; ++s){
            for (int a = 0; a < mdpInstance.nActions; ++a){
                for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                    PBar[s][a][sPlus] += mu * (PNew[s][a][sPlus] - PBar[s][a][sPlus]);
                }
            }
        }
        JIter = inner_product(valueNew, mdpInstance.rho);
        double PhiVal = PHIVal(mdpInstance, policy, PNew, kappa, stepP, max_iter);
        if (t % 5 == 0){
            cout << "t:" << t << " JIter:" << JIter << ", Phi Val:"<< PhiVal << "\n";
        }
        JHis.push_back(JIter);
        phiHis.push_back(PhiVal);
        if (t == max_iter){
            return {policy, valueNew, JHis, phiHis};
        }
    }
}


//void robustValueIterAlgGarnet(
//    const int& garnetSNum,
//    const int& garnetANum,
//    const int& garnetBNum,
//    const int& maxIter,
//    const int& randomSeedNum
//){
//    string dataSetName = "garnetSA_" + to_string(garnetSNum) + "_" + to_string(garnetANum) + "_" + to_string(garnetBNum) + "_";
//    vector<vector<double>> robustValueJHis; robustValueJHis.reserve(randomSeedNum);
//    // robust value iteration
//    for (int seed = 0; seed < randomSeedNum; ++seed){
//        string fileName = "./res/robustValueIteration_"+ dataSetName + to_string(seed) + "_J_SA.csv";
//        ofstream ofs(fileName, ofstream::out);
//        auto start = std::chrono::steady_clock::now();
//        MDP mdpInstance = garnet_mdps_sa(garnetSNum, garnetANum, garnetBNum, seed + 2);
//        // mdpInstance.view();
//        for (int policySeed = 1; policySeed <= 10; ++policySeed){
//            vector<vector<double>> policyIni = random_policy(mdpInstance, policySeed);
//            vector<vector<double>> kappa = random_tolerance_kappa_sa(mdpInstance, policySeed);
//            vector<double> valueIni = mdp_value(mdpInstance, policyIni, mdpInstance.P);
//            auto result = robustValueIteration(mdpInstance, kappa, maxIter, valueIni);
//            auto policy = std::get<0>(result);
//            auto v = std::get<1>(result);
//            auto J = std::get<2>(result);
//            auto end = std::chrono::steady_clock::now();
//            std::chrono::duration<double> elapsed_seconds = end - start;
//            cout<< "#######------robust value iteration(" << seed << ")" << "policy seed(" << policySeed << ")"<< " elapsed time: " << elapsed_seconds.count() << "s\n";
//            for (int j = 0; j < maxIter; ++j){
//                ofs << J[j] << ",";
//            }
//            ofs << "\n";
//        }
//        ofs.close();
//    }
//}


void drpgAlgGarnet_sa(
    const int& garnetSNum,
    const int& garnetANum,
    const int& garnetBNum,
    const double &outerStepSize,
    const double &innerStepSize,
    const int& maxIter,
    const int& randomSeedNum, 
    const int& maxIterAll
){
    // drpg method
    string dataSetName = "garnetSA_" + to_string(garnetSNum) + "_" + to_string(garnetANum) + "_" + to_string(garnetBNum) + "_";
    for (int seed = 0; seed < randomSeedNum; ++seed){
        auto start = std::chrono::steady_clock::now();
        MDP mdpInstance = garnet_mdps_sa(garnetSNum, garnetANum, garnetBNum, seed + 2);
        // mdpInstance.view();
        vector<vector<double>> kappa = random_tolerance_kappa_sa(mdpInstance, seed + 2);
        string fileNameDRPG = "./res/drpg_"+ dataSetName + to_string(seed) + "_J_SA.csv";
        ofstream ofsDRPG(fileNameDRPG, ofstream::out);
        string fileNameDRPGPhi = "./res/drpg_"+ dataSetName + to_string(seed) + "_Phi_SA.csv";
        ofstream ofsDRPGPhi(fileNameDRPGPhi, ofstream::out);
        string fileNameDRPGRecord = "./res/drpg_"+ dataSetName + to_string(seed) + "_Record_SA.csv";
        ofstream ofsDRPGRecord(fileNameDRPGRecord, ofstream::out);
        for (int policySeed = 1; policySeed <= 10; ++policySeed){
            auto result = drpg_sa(mdpInstance, kappa, outerStepSize, innerStepSize, maxIter, maxIterAll, policySeed);
            auto policy = std::get<0>(result);
            auto v = std::get<1>(result);
            auto J = std::get<2>(result);
            auto phiVal = std::get<3>(result);
            auto recordBreak = std::get<4>(result);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            cout<< "#######------drpg method(" << seed << ")"<< "elapsed time: " << elapsed_seconds.count() << "s\n";
            for (int j = 0; j < maxIterAll; ++j){
                ofsDRPG << J[j] << ",";
            }
            ofsDRPG << "\n";
            for (int j = 0; j < maxIterAll; ++j){
                ofsDRPGPhi << phiVal[j] << ",";
            }
            ofsDRPGPhi << "\n";
            for (const auto& record : recordBreak){
                ofsDRPGRecord << record << ",";
            }
            ofsDRPGRecord << "\n";
        }
        ofsDRPG.close();
        ofsDRPGPhi.close();
        ofsDRPGRecord.close();
    }
}

void dsgdaAlgGarnet_sa(
   const int& garnetSNum,
   const int& garnetANum,
   const int& garnetBNum,
   const double &stepPolicy,
   const double &stepP,
   const int& maxIter,
   const double& r1,
   const double& r2,
   const double& beta,
   const double& mu,
   const int& randomSeedNum
){
    string dataSetName = "garnetSA_" + to_string(garnetSNum) + "_" + to_string(garnetANum) + "_" + to_string(garnetBNum) + "_";
    // dsgda method
    for (int seed = 0; seed < randomSeedNum; ++seed){
        auto start = std::chrono::steady_clock::now();
        MDP mdpInstance = garnet_mdps_sa(garnetSNum, garnetANum, garnetBNum, seed + 2);
        // mdpInstance.view();
        vector<vector<double>> kappa = random_tolerance_kappa_sa(mdpInstance, seed + 2);
        string fileNameDSGDA = "./res/dsgda_" + dataSetName + to_string(seed) + "_J_SA.csv";
        ofstream ofsDSGDA(fileNameDSGDA, ofstream::out);
        string fileNameDSGDAPhi = "./res/dsgda_" + dataSetName + to_string(seed) + "_Phi_SA.csv";
        ofstream ofsDSGDAPhi(fileNameDSGDAPhi, ofstream::out);
        for (int policySeed = 1; policySeed <= 10; ++policySeed){
            auto result = dsgda_sa(mdpInstance, kappa, stepPolicy, stepP, maxIter, beta, mu, r1, r2, policySeed);
            auto policy = std::get<0>(result);
            auto v = std::get<1>(result);
            auto J = std::get<2>(result);
            auto phiVal = std::get<3>(result);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            cout<< "#######------dsgda method(" << seed << ")" << "policySeed:" << policySeed << " elapsed time: " << elapsed_seconds.count() << "s\n";
            for (int j = 0; j < maxIter; ++j){
                ofsDSGDA << J[j] << ",";
            }
            ofsDSGDA << "\n";
            for (int j = 0; j < maxIter; ++j){
                ofsDSGDAPhi << phiVal[j] << ",";
            }
            ofsDSGDAPhi << "\n";
        }
        ofsDSGDA.close();
        ofsDSGDAPhi.close();
    }
}

void garnetSA_5_6_3(){
    int stateNum = 5; // 10, 15
    int actionNum = 6;
    int branchNum = 3;
    double outerStepSize = 0.05;
    double innerStepSize = 0.1;
    int maxIter = 200;
    int randomSeedNum = 2;
    int maxIterAll = 1000;
//    robustValueIterAlgGarnet(stateNum, actionNum, branchNum, maxIter, randomSeedNum);
    maxIter = 100000; //large enough here, we will stop by maxIterAll
    drpgAlgGarnet_sa(stateNum, actionNum, branchNum, outerStepSize, innerStepSize, maxIter, randomSeedNum, maxIterAll);
    double stepPolicy = 0.05; // 0.0003;
    double stepP = 0.1; // 0.00003;
    maxIter = maxIterAll / 2;
    double r1 = 16.0;
    double r2 = 12.0;
    double beta = 0.3; // 0.05
    double mu = 0.3; // 0.05
    dsgdaAlgGarnet_sa(stateNum, actionNum, branchNum, stepPolicy, stepP, maxIter, r1, r2, beta, mu, randomSeedNum);
}

void garnetSA_10_5_10(){
    int stateNum = 10;
    int actionNum = 5;
    int branchNum = 10;
    double outerStepSize = 0.05;
    double innerStepSize = 0.1;
    int maxIter = 200;
    int randomSeedNum = 2;
    int maxIterAll = 1000;
//   robustValueIterAlgGarnet(stateNum, actionNum, branchNum, maxIter, randomSeedNum);
    maxIter = 100000; //large enough here, we will stop by maxIterAll
    drpgAlgGarnet_sa(stateNum, actionNum, branchNum, outerStepSize, innerStepSize, maxIter, randomSeedNum, maxIterAll);
    double stepPolicy = 0.05; // 0.0003;
    double stepP = 0.1; // 0.00003;
    maxIter = maxIterAll / 2;
    double r1 = 16.0;
    double r2 = 12.0;
    double beta = 0.3;
    double mu = 0.3;
    dsgdaAlgGarnet_sa(stateNum, actionNum, branchNum, stepPolicy, stepP, maxIter, r1, r2, beta, mu, randomSeedNum);
}

