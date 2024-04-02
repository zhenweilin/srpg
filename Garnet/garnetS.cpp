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
#include <thread>
#include "gurobi_c++.h"
#include <sstream>
#include "Eigen/Dense"
using namespace Eigen;

vector<int> P_branch(size_t nStates, size_t nBranchs, int seed) {
    /*
    select nBranchs size state from nStates
    */
    srand(unsigned(time(NULL)));
    vector<int> S_temp; S_temp.reserve(nStates);
    vector<int> b_temp; b_temp.reserve(nBranchs);
    for (size_t s = 0; s < nStates; ++s) {
        int s_temp = (int) s;
        S_temp.push_back(s_temp);
    }
    std::shuffle(S_temp.begin(), S_temp.end(), default_random_engine(seed));
    for (size_t b = 0; b < nBranchs; b++) {
        b_temp.push_back(S_temp[b]);
    }
    return b_temp;
}

MDP garnet_mdps(size_t nStates, size_t nActions, size_t nBranchs, int seed){
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

vector<double> random_tolerance_kappa(const MDP& mdpInstance, int seed){
    /*
    generate random tolerance kappa
    */
    vector<double> kappa; kappa.reserve(mdpInstance.nStates);
    default_random_engine generator;
    generator.seed(seed);
    uniform_real_distribution<double> distribution(0.1, 0.5);
    for (size_t s = 0; s < mdpInstance.nStates; ++s){
        kappa.push_back(distribution(generator));
    }
    return kappa;
}


/***************************************************************
DRPG for S-rectangularity RMDP (pi, v_opt)
**************************************************************/

vector<vector<double>> random_policy(const MDP& mdpInstance, int seed){
    /*
    generate random policy
    */
    vector<vector<double>> pi; pi.reserve(mdpInstance.nStates);
    default_random_engine generator;
    generator.seed(seed);
    uniform_real_distribution<double> distribution(0.1, 0.5);
    for (size_t s = 0; s < mdpInstance.nStates; ++s){
        vector<double> pi_s; pi_s.reserve(mdpInstance.nActions);
        double pi_sSum = 0.0;
        for (size_t a = 0; a < mdpInstance.nActions; ++a){
            double pi_sa = distribution(generator);
            pi_sSum += pi_sa;
            pi_s.push_back(pi_sa);
        }
        for (size_t a = 0; a < mdpInstance.nActions; ++a){
            pi_s[a] /= pi_sSum;
        }
        pi.push_back(pi_s);
    }
    return pi;
}

vector<vector<vector<double>>> random_P(const MDP& mdpInstance, int seed){
    /*generate random P*/
    vector<vector<vector<double>>> P; P.reserve(mdpInstance.nStates);
    default_random_engine generator;
    generator.seed(seed);
    uniform_real_distribution<double> distribution(0.1, 0.5);
    for (size_t s = 0; s < mdpInstance.nStates; ++s){
        vector<vector<double>> P_s; P_s.reserve(mdpInstance.nActions);
        for (size_t a = 0; a < mdpInstance.nActions; ++a){
            vector<double> P_sa; P_sa.reserve(mdpInstance.nStates);
            double P_saSum = 0.0;
            for (size_t sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                double P_sas = distribution(generator);
                P_saSum += P_sas;
                P_sa.push_back(P_sas);
            }
            for (size_t sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                P_sa[sPlus] /= P_saSum;
            }
            P_s.push_back(P_sa);
        }
        P.push_back(P_s);
    }
    return P;
}

vector<double> mdp_occupancy(
    const MDP& mdpInstance,
    const vector<vector<vector<double>>>& PNow,
    const vector<vector<double>>& policyNow
){
    /*calculate the occupancy measure*/
    // initialize probability under policy now
    vector<vector<double>> P_pi; P_pi.reserve(mdpInstance.nStates);
    for(int s = 0; s < mdpInstance.nStates; ++s){
        vector<double> P_pi_s; P_pi_s.reserve(mdpInstance.nStates);
        for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
            double temp = 0.0;
            for (int a = 0; a < mdpInstance.nActions; ++a){
                temp += policyNow[s][a] * PNow[s][a][sPlus];
            }
            P_pi_s.push_back(temp);
        }
        P_pi.push_back(P_pi_s);
    }
    // construt P_pi matrix
    MatrixXd P_pi_matrix(mdpInstance.nStates, mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
            P_pi_matrix(s, sPlus) = P_pi[s][sPlus];
        }
    }
    // Identity matrix
    MatrixXd I = MatrixXd::Identity(mdpInstance.nStates, mdpInstance.nStates);

    // I - gamma * P_pi
    MatrixXd I_gamma_P_pi = I - mdpInstance.gamma * P_pi_matrix;

    // initialize state distribution * (1 - gamma)
    VectorXd rhoGamma(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        rhoGamma(s) = mdpInstance.rho[s] * (1 - mdpInstance.gamma);
    }
    VectorXd occupancy = I_gamma_P_pi.colPivHouseholderQr().solve(rhoGamma);
    vector<double> eta; eta.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        eta.push_back(occupancy(s));
    }
    return eta;
}

vector<double> mdp_value(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P
){
    /*calculate the value function for fixed policy and p*/
    MatrixXd I = MatrixXd::Identity(mdpInstance.nStates, mdpInstance.nStates);
    // calculate c_pi
    vector<double> c; c.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        double temp = 0.0;
        for (int a = 0; a < mdpInstance.nActions; ++a){
            double temp2 = 0.0;
            for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                temp2 += P[s][a][sPlus] * mdpInstance.cost[s][a][sPlus];
            }
            temp += policy[s][a] * temp2;
        }
        c.push_back(temp);
    }
    VectorXd c_pi(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        c_pi(s) = c[s];
    }
    // calculate P_pi
    vector<vector<double>> P_pi; P_pi.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        vector<double> P_pi_s; P_pi_s.reserve(mdpInstance.nStates);
        for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
            double temp = 0.0;
            for (int a = 0; a < mdpInstance.nActions; ++a){
                temp += policy[s][a] * P[s][a][sPlus];
            }
            P_pi_s.push_back(temp);
        }
        P_pi.push_back(P_pi_s);
    }
    MatrixXd P_pi_matrix(mdpInstance.nStates, mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
            P_pi_matrix(s, sPlus) = P_pi[s][sPlus];
        }
    }

    MatrixXd I_gamma_P_pi = I - mdpInstance.gamma * P_pi_matrix;
    VectorXd v_pi = I_gamma_P_pi.colPivHouseholderQr().solve(c_pi);
    vector<double> v; v.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        v.push_back(v_pi(s));
    }
    return v;
}

vector<vector<vector<double>>> MDP_grad_P(
    const MDP& mdpInstance,
    const vector<vector<double>> &policy,
    const vector<vector<vector<double>>> &P,
    const vector<double>& eta,
    const vector<double>& v_now
){
    /* calculate the gradient of transition kernel */
    vector<vector<vector<double>>> grad_P; grad_P.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        vector<vector<double>> grad_P_s; grad_P_s.reserve(mdpInstance.nActions);
        for (int a = 0; a < mdpInstance.nActions; ++a){
            vector<double> grad_P_sa; grad_P_sa.reserve(mdpInstance.nStates);
            double temp_c;
            double grad;
            for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                temp_c = mdpInstance.cost[s][a][sPlus] + mdpInstance.gamma * v_now[sPlus];
                grad = (1 / (1 - mdpInstance.gamma)) * eta[s] * policy[s][a] * temp_c;
                grad_P_sa.push_back(grad);
            }
            grad_P_s.push_back(grad_P_sa);
        }
        grad_P.push_back(grad_P_s);
    }
    return grad_P;
}

vector<vector<vector<double>>> dsgda_grad_P(
    const MDP& mdpInstance,
    const vector<vector<double>> &policy,
    const vector<vector<vector<double>>> &P,
    const vector<vector<vector<double>>> &Pbar,
    const double r2,
    const vector<double>& eta,
    const vector<double>& v_now
){
    /* calculate the gradient of transition kernel */
    vector<vector<vector<double>>> grad_P; grad_P.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        vector<vector<double>> grad_P_s; grad_P_s.reserve(mdpInstance.nActions);
        for (int a = 0; a < mdpInstance.nActions; ++a){
            vector<double> grad_P_sa; grad_P_sa.reserve(mdpInstance.nStates);
            double temp_c;
            double grad;
            for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                temp_c = mdpInstance.cost[s][a][sPlus] + mdpInstance.gamma * v_now[sPlus];
                grad = (1 / (1 - mdpInstance.gamma)) * eta[s] * policy[s][a] * temp_c - r2 * (P[s][a][sPlus] - Pbar[s][a][sPlus]);
                grad_P_sa.push_back(grad);
            }
            grad_P_s.push_back(grad_P_sa);
        }
        grad_P.push_back(grad_P_s);
    }
    return grad_P;
}

void updateP_one(
                 const MDP& mdpInstance,
                 const vector<vector<double>>& policy,
                 vector<vector<vector<double>>>& PNew,
                 const vector<double>& kappa,
                 vector<vector<vector<double>>> grad_P,
                 const double& stepsize,
                 int s
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
    GRBVar *p_s;
    p_s = model.addVars(nActions * nStates, GRB_CONTINUOUS);
    for (int a = 0; a < nActions; ++a){
        // Define decision variables
        for (int sPlus = 0; sPlus < nStates; ++sPlus){
            p_s[a * nStates + sPlus].set(GRB_DoubleAttr_LB, 0.0);
            p_s[a * nStates + sPlus].set(GRB_DoubleAttr_UB, 1.0);
        }
    }
    
    GRBVar* temp_y;
    temp_y = model.addVars(nActions * nStates, GRB_CONTINUOUS);
    for (int a = 0; a < nActions; ++a){
        for (int sPlus = 0; sPlus < nStates; ++sPlus){
            temp_y[a * nStates + sPlus].set(GRB_DoubleAttr_LB, -inf);
            temp_y[a * nStates + sPlus].set(GRB_DoubleAttr_UB, inf);
        }
    }
    
    
    // expression
    GRBQuadExpr obj;
    vector<GRBLinExpr> sum_p; sum_p.reserve(nActions);
    vector<GRBLinExpr> sum_bp; sum_bp.reserve(nActions);
    for (int a = 0; a < nActions; ++a){
        // p_sa^T 1 = 1
        for (int sPlus = 0; sPlus < nStates; ++sPlus){
            sum_p[a] += p_s[a * nStates + sPlus];
        }
        model.addConstr(sum_p[a] == 1.0);

        // branch location constraint
        for (int sPlus = 0; sPlus < nStates; ++sPlus){
            sum_bp[a] += B[s][a][sPlus] * p_s[a * nStates + sPlus];
        }
        model.addConstr(sum_bp[a] == 0.0);
        // set objective: minimize -<grad, p> + 1/2 * stepsize * ||p - PNew||^2
        for (int sPlus = 0; sPlus < nStates; ++sPlus){
            obj += (p_s[a * nStates + sPlus] - PNew[s][a][sPlus]) * (p_s[a * nStates + sPlus] - PNew[s][a][sPlus]);
            obj -= 2 * stepsize * grad_P[s][a][sPlus] * p_s[a * nStates + sPlus];
        }
    }
    GRBLinExpr sum_y;
    // sum_y <= kappa[s]
    for (int a = 0; a < nActions; ++a){
        for (int sPlus = 0; sPlus < nStates; ++sPlus){
            sum_y += temp_y[a * nStates + sPlus];
            model.addConstr(temp_y[a * nStates + sPlus] >= p_s[a * nStates + sPlus] - P[s][a][sPlus]);
            model.addConstr(temp_y[a * nStates + sPlus] >= P[s][a][sPlus] - p_s[a * nStates + sPlus]);
        }
    }
    model.addConstr(sum_y <= kappa[s]);
    
    
    model.setObjective(obj, GRB_MINIMIZE);
    // Optimize model
    model.optimize();
    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL){
        for (int a = 0; a < nActions; ++a){
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                PNew[s][a][sPlus] = p_s[a * nStates + sPlus].get(GRB_DoubleAttr_X);
            }
        }
    }
    else{
        cout << "Error: optimization failed" << endl;
        exit(1);
    }
}

vector<vector<vector<double>>> updateP_multi(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    vector<vector<vector<double>>>& PNew,
    const vector<double>& kappa,
    vector<vector<vector<double>>> grad_P,
    const double& stepsize
                                       ){
    int nStates = (int)mdpInstance.nStates;
    int nActions = (int)mdpInstance.nActions;
    const vector<vector<vector<double>>>& P = mdpInstance.P;
    const vector<vector<vector<double>>>& B = mdpInstance.B;
    const vector<vector<vector<double>>>& cost = mdpInstance.cost;
    double inf = numeric_limits<double>::infinity();
    std::vector<std::thread> threads;
    for (int s = 0; s < nStates; ++s){
        threads.emplace_back(updateP_one, std::ref(mdpInstance), std::ref(policy), std::ref(PNew), std::ref(kappa), std::ref(grad_P), stepsize, s);
    }
    
    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    return PNew;
}


vector<vector<vector<double>>> updateP(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    vector<vector<vector<double>>> PNew,
    const vector<double>& kappa,
    vector<vector<vector<double>>> grad_P,
    const double& stepsize
){
    /*
     Change into S rectangular
     */
    int nStates = (int)mdpInstance.nStates;
    int nActions = (int)mdpInstance.nActions;
    const vector<vector<vector<double>>>& P = mdpInstance.P;
    const vector<vector<vector<double>>>& B = mdpInstance.B;
    const vector<vector<vector<double>>>& cost = mdpInstance.cost;
    double inf = numeric_limits<double>::infinity();
    for (int s = 0; s < nStates; ++s){
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 0);
        env.start();
        GRBModel model = GRBModel(env);
        GRBVar *p_s;
        p_s = model.addVars(nActions * nStates, GRB_CONTINUOUS);
        for (int a = 0; a < nActions; ++a){
            // Define decision variables
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                p_s[a * nStates + sPlus].set(GRB_DoubleAttr_LB, 0.0);
                p_s[a * nStates + sPlus].set(GRB_DoubleAttr_UB, 1.0);
            }
        }
        
        GRBVar* temp_y;
        temp_y = model.addVars(nActions * nStates, GRB_CONTINUOUS);
        for (int a = 0; a < nActions; ++a){
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                temp_y[a * nStates + sPlus].set(GRB_DoubleAttr_LB, -inf);
                temp_y[a * nStates + sPlus].set(GRB_DoubleAttr_UB, inf);
            }
        }
        
        
        // expression
        GRBQuadExpr obj;
        vector<GRBLinExpr> sum_p; sum_p.reserve(nActions);
        vector<GRBLinExpr> sum_bp; sum_bp.reserve(nActions);
        for (int a = 0; a < nActions; ++a){
            // p_sa^T 1 = 1
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_p[a] += p_s[a * nStates + sPlus];
            }
            model.addConstr(sum_p[a] == 1.0);

            // branch location constraint
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_bp[a] += B[s][a][sPlus] * p_s[a * nStates + sPlus];
            }
            model.addConstr(sum_bp[a] == 0.0);
            // set objective: minimize -<grad, p> + 1/2 * stepsize * ||p - PNew||^2
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                obj += (p_s[a * nStates + sPlus] - PNew[s][a][sPlus]) * (p_s[a * nStates + sPlus] - PNew[s][a][sPlus]);
                obj -= 2 * stepsize * grad_P[s][a][sPlus] * p_s[a * nStates + sPlus];
            }
        }
        GRBLinExpr sum_y;
        // sum_y <= kappa[s]
        for (int a = 0; a < nActions; ++a){
            for (int sPlus = 0; sPlus < nStates; ++sPlus){
                sum_y += temp_y[a * nStates + sPlus];
                model.addConstr(temp_y[a * nStates + sPlus] >= p_s[a * nStates + sPlus] - P[s][a][sPlus]);
                model.addConstr(temp_y[a * nStates + sPlus] >= P[s][a][sPlus] - p_s[a * nStates + sPlus]);
            }
        }
        model.addConstr(sum_y <= kappa[s]);
        
        
        model.setObjective(obj, GRB_MINIMIZE);
        // Optimize model
        model.optimize();
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL){
            for (int a = 0; a < nActions; ++a){
                for (int sPlus = 0; sPlus < nStates; ++sPlus){
                    PNew[s][a][sPlus] = p_s[a * nStates + sPlus].get(GRB_DoubleAttr_X);
                }
            }
        }
        else{
            cout << "Error: optimization failed" << endl;
            exit(1);
        }
    }
    return {PNew};
}


double PHIVal(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P_0,
    const vector<double>& kappa,
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
    vector<vector<vector<double>>> PTemp;
    vector<vector<vector<double>>> PNew = P_0;
    PTemp.reserve(nStates);
    for (size_t s = 0; s < nStates; ++s){
        vector<vector<double>> PTemp_s; PTemp_s.reserve(nActions);
        for (size_t a = 0; a < nActions; ++a){
            vector<double> PTemp_sa(nStates, 0.0);
            PTemp_s.push_back(PTemp_sa);
        }
        PTemp.push_back(PTemp_s);
    }
    while (true){
        t += 1;
        valueNew = mdp_value(mdpInstance, policy, PNew);
        vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
        vector<vector<vector<double>>> grad_P = MDP_grad_P(mdpInstance, policy, PNew, eta, valueNew);
        PTemp = updateP_multi(mdpInstance, policy, PNew, kappa, grad_P, 1.0);
        double gap = 0.0;
        for (size_t sPlus = 0; sPlus < nStates; ++sPlus){
            gap += (valueNew[sPlus] - valueOld[sPlus]) * (valueNew[sPlus] - valueOld[sPlus]);
        }
        gap = sqrt(gap);
        if (gap < 1e-5 || t == max_iter){
            if (t == max_iter){
                cout << "Warning: maximum iteration reached" << endl;
            }
            double PHI = inner_product(valueNew, rho);
            return PHI;
        }
        PNew = PTemp;
        valueOld = valueNew;
    }
}



tuple<vector<vector<vector<double>>>, vector<double>, int> drpg_inner_loop_pgd(
    const MDP& mdpInstance, 
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P_0,
    const vector<double>& kappa,
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
    vector<vector<vector<double>>> PNew = P_0;
    double JIter = 0.0;
    while (true){
        t += 1;
        valueNew = mdp_value(mdpInstance, policy, PNew);
        vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
        vector<vector<vector<double>>> grad_P = MDP_grad_P(mdpInstance, policy, PNew, eta, valueNew);
        PNew = updateP_multi(mdpInstance, policy, PNew, kappa, grad_P, stepsize);
        double gap = 0.0;
        for (size_t sPlus = 0; sPlus < nStates; ++sPlus){
            gap += (valueNew[sPlus] - valueOld[sPlus]) * (valueNew[sPlus] - valueOld[sPlus]);
        }
        gap = sqrt(gap);
        JIter = inner_product(valueNew, rho);
        JHis.push_back(JIter);
        PhiHis.push_back(phiValOuter);
        if (gap < epsilon || t == max_iter){
            return {PNew, valueNew, t};
            break;
        }
        valueOld = valueNew;
    }
}




vector<vector<double>> mdp_grad_policy(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P,
    const vector<double>& eta,
    const vector<double>& v_now
){
    vector<vector<double>> grad_policy; grad_policy.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        vector<double> grad_policy_s; grad_policy_s.reserve(mdpInstance.nActions);
        for (int a = 0; a < mdpInstance.nActions; ++a){
            vector<double> temp_c; temp_c.reserve(mdpInstance.nStates);
            for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                temp_c.push_back(mdpInstance.cost[s][a][sPlus] + mdpInstance.gamma * v_now[sPlus]);
            }
            double q_sa = 0;
            for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                q_sa += P[s][a][sPlus] * temp_c[sPlus];
            }
            double temp_grad_sa = (1 / (1 - mdpInstance.gamma)) * eta[s] * q_sa;
            grad_policy_s.push_back(temp_grad_sa);
        }
        grad_policy.push_back(grad_policy_s);
    }
    return grad_policy;
}

vector<vector<double>> dsgda_grad_policy(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<double>>& policyBar,
    const double r1,
    const vector<vector<vector<double>>>& P,
    const vector<double>& eta,
    const vector<double>& v_now
){
    vector<vector<double>> grad_policy; grad_policy.reserve(mdpInstance.nStates);
    for (int s = 0; s < mdpInstance.nStates; ++s){
        vector<double> grad_policy_s; grad_policy_s.reserve(mdpInstance.nActions);
        for (int a = 0; a < mdpInstance.nActions; ++a){
            vector<double> temp_c; temp_c.reserve(mdpInstance.nStates);
            for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                temp_c.push_back(mdpInstance.cost[s][a][sPlus] + mdpInstance.gamma * v_now[sPlus]);
            }
            double q_sa = 0;
            for (int sPlus = 0; sPlus < mdpInstance.nStates; ++sPlus){
                q_sa += P[s][a][sPlus] * temp_c[sPlus];
            }
            double temp_grad_sa = (1 / (1 - mdpInstance.gamma)) * eta[s] * q_sa + r1 * (policy[s][a] - policyBar[s][a]);
            grad_policy_s.push_back(temp_grad_sa);
        }
        grad_policy.push_back(grad_policy_s);
    }
    return grad_policy;
}

void update_policy_one(const MDP& mdpInstance,
                       const vector<vector<double>>& policy,
                       const double& stepsize,
                       const vector<vector<double>>& grad_policy,
                       vector<vector<double>>& policy_new,
                       int s){
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

vector<vector<double>> update_policy_multi(
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
    std::vector<std::thread> threads;
    for (int s = 0; s < mdpInstance.nStates; ++s){
        threads.emplace_back(update_policy_one, std::ref(mdpInstance), std::ref(policy), stepsize, std::ref(grad_policy), std::ref(policy_new), s);
    }
    
    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    return policy_new;
}


vector<vector<double>> update_policy(
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

double nrm2P(const vector<vector<vector<double>>> &P){
    double norm2 = 0.0;
    for (const auto& P_s : P){
        for (const auto& P_sa : P_s){
            for (const auto& P_sas : P_sa){
                norm2 += P_sas * P_sas;
            }
        }
    }
    return sqrt(norm2);
}

double nrm2Policy(const vector<vector<double>>& policy){
    double norm2 = 0.0;
    for (const auto& policy_s : policy){
        for (const auto& policy_sa : policy_s){
            norm2 += policy_sa * policy_sa;
        }
    }
    return sqrt(norm2);
}

// Compute l2-norm for vector library
double L2_Gap(const vector<double>& v1, const vector<double>& v2)
{
    int S = (int) v1.size();
    vector<double> v_gap;
    for (int i_1 = 0; i_1 < S; i_1++)
    {
        v_gap.push_back(v1[i_1] - v2[i_1]);
    }
    double sum1 = 0.0;
    for (int i_2 = 0; i_2 < S; i_2++)
    {
        sum1 += v_gap[i_2] * v_gap[i_2];
    }
    double gap = sqrt(sum1);
    return gap;
}
extern double inner_product(const vector<double>& v1, const vector<double>& v2)
{
    int S = (int) v1.size();
    double x = 0.0;
    for (int i = 0; i < S; i++)
    {
        x += v1[i] * v2[i];
    }
    return x;
}


tuple<vector<vector<double>>, vector<double>, vector<double>, vector<double>, vector<int>> drpg_s(
    const MDP& mdpInstance,
    const vector<double>& kappa,
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
        policy = update_policy_multi(mdpInstance, policy, stepOuter, grad_policy);
        JIter = inner_product(v, mdpInstance.rho);
        PhiValOuter = PHIVal(mdpInstance, policy, P, kappa, stepInner, max_iter);
        if (t % 5 == 0){
            cout<< "drpg method " << "t:" << t << " JIter:" << JIter << ", Phi Val:"<< PhiValOuter << "\n";
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

tuple<vector<vector<double>>, vector<double>, vector <double>, vector <double>> dsgda_s(
    const MDP& mdpInstance,
    const vector<double>& kappa,
    double stepPolicy,
    double stepP,
    const int& max_iter,
    const double& beta,
    const double& mu,
    const double& r1,
    const double& r2,
    const double& policySeed
){
    // random policy
    vector<vector<double>> policy = random_policy(mdpInstance, policySeed);
    vector<vector<vector<double>>> P = mdpInstance.P;
    vector<vector<vector<double>>> PNew = P;
    vector<vector<double>> policyBar = policy;
    vector<vector<vector<double>>> PBar = random_P(mdpInstance, policySeed);
    int t = 0;
    vector<vector<double>> grad_policy;
    vector<vector<vector<double>>> grad_P;
    vector<double> JHis; JHis.reserve(max_iter);
    vector<double> phiHis; phiHis.reserve(max_iter);
    double JIter;
    auto valueNew = mdp_value(mdpInstance, policy, PNew);
    vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
    while (true)
    {
        t += 1;
        grad_policy = dsgda_grad_policy(mdpInstance, policy, policyBar, r1, PNew, eta, valueNew);
        policy = update_policy_multi(mdpInstance, policy, stepPolicy, grad_policy);

        valueNew = mdp_value(mdpInstance, policy, PNew);
        vector<double> eta = mdp_occupancy(mdpInstance, PNew, policy);
        grad_P = dsgda_grad_P(mdpInstance, policy, PNew, PBar, r2, eta, valueNew);
        PNew = updateP_multi(mdpInstance, policy, PNew, kappa, grad_P, stepP);

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
        if (t % 1 == 0){
            cout << "t:" << t << " JIter:" << JIter << ", Phi Val:"<< PhiVal << "\n";
        }
        JHis.push_back(JIter);
        phiHis.push_back(PhiVal);
        if (t == max_iter){
            return {policy, valueNew, JHis, phiHis};
        }
    }
}



void drpgAlgGarnet(
    const int& garnetSNum,
    const int& garnetANum,
    const int& garnetBNum,
    const double &outerStepSize,
    const double &innerStepSize,
    const int& maxIter,
    const int& randomSeedNum, 
    const int& maxIterAll
){
    string dataSetName = "garnet_" + to_string(garnetSNum) + "_" + to_string(garnetANum) + "_" + to_string(garnetBNum) + "_";
    // drpg method
    for (int seed = 0; seed < randomSeedNum; ++seed){
        auto start = std::chrono::steady_clock::now();
        MDP mdpInstance = garnet_mdps(garnetSNum, garnetANum, garnetBNum, seed + 2);
        // mdpInstance.view();
        vector<double> kappa = random_tolerance_kappa(mdpInstance, seed + 2);
        string fileNameDRPG= "./res/drpg_" + dataSetName + to_string(seed) + "_J_S.csv";
        ofstream ofsDRPG(fileNameDRPG, ofstream::out);
        string fileNameDRPGPhi = "./res/drpg_" + dataSetName + to_string(seed) + "_Phi_S.csv";
        ofstream ofsDRPGPhi(fileNameDRPGPhi, ofstream::out);
        string fileNameDRPGRecord = "./res/drpg_" + dataSetName + to_string(seed) + "_" + "Record_S.csv";
        ofstream ofsDRPGRecord(fileNameDRPGRecord, ofstream::out);
        for (int policySeed = 1; policySeed <= 10; ++policySeed){
            auto result = drpg_s(mdpInstance, kappa, outerStepSize, innerStepSize, maxIter, maxIterAll, policySeed);
            auto policy = std::get<0>(result);
            auto v = std::get<1>(result);
            auto J = std::get<2>(result);
            auto phiVal = std::get<3>(result);
            auto recordBreak = std::get<4>(result);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            cout<< "#######------drpg method(" << seed << ") policy Seed:"<< policySeed <<" elapsed time: " << elapsed_seconds.count() << "s\n";
            for (int j = 0; j < maxIterAll; ++j)
            {
                ofsDRPG << J[j] << ",";
            }
            ofsDRPG << "\n";
            cout << "maxIterAll:" << maxIterAll << "\n";
            cout << phiVal.size() << "\n";
            for (int i = 0; i < maxIterAll; ++i){
                ofsDRPGPhi << phiVal[i] << ",";
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

void dsgdaAlgGarnet(
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
    string dataSetName = "garnet_" + to_string(garnetSNum) + "_" + to_string(garnetANum) + "_" + to_string(garnetBNum) + "_";
    // dsgda method
    for (int seed = 0; seed < randomSeedNum; ++seed){
        MDP mdpInstance = garnet_mdps(garnetSNum, garnetANum, garnetBNum, seed + 2);
        // mdpInstance.view();
        vector<double> kappa = random_tolerance_kappa(mdpInstance, seed + 2);
        string fileNameDSGDA = "./res/dsgda_" + dataSetName + to_string(seed) + "_J_S.csv" ;
        ofstream ofsDSGDA(fileNameDSGDA, ofstream::out);
        string fileNameDSGDAPhi = "./res/dsgda_" + dataSetName + to_string(seed) + "_Phi_S.csv";
        ofstream ofsDSGDAPhi(fileNameDSGDAPhi, ofstream::out);
        for (int policySeed = 1; policySeed <= 10; ++policySeed ){
            auto start = std::chrono::steady_clock::now();
            auto result = dsgda_s(mdpInstance, kappa, stepPolicy, stepP, maxIter, beta, mu, r1, r2, policySeed);
            auto policy = std::get<0>(result);
            auto v = std::get<1>(result);
            auto J = std::get<2>(result);
            auto phiVal = std::get<3>(result);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            cout<< "#######------dsgda method(" << seed << ") policySeed:"<< policySeed << " elapsed time: " << elapsed_seconds.count() << "s\n";
            for (int j = 0; j < maxIter; ++j)
            {
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


void garnet_5_6_3(){
    int stateNum = 5; // 10, 15
    int actionNum = 6;
    int branchNum = 3;
    double outerStepSize = 0.05;
    double innerStepSize = 0.1;
    int maxIter = 200;
    int randomSeedNum = 2;
    int maxIterAll = 1000;
    maxIter = 100000; //large enough here, we will stop by maxIterAll
    drpgAlgGarnet(stateNum, actionNum, branchNum, outerStepSize, innerStepSize, maxIter, randomSeedNum, maxIterAll);
    double stepPolicy = 0.05; // 0.0003;
    double stepP = 0.05; // 0.00003;
    maxIter = maxIterAll / 2;
    double r1 = 32.0;
    double r2 = 24.0;
    double beta = 0.4;
    double mu = 0.4;
    dsgdaAlgGarnet(stateNum, actionNum, branchNum, stepPolicy, stepP, maxIter, r1, r2, beta, mu, randomSeedNum);
}

void garnet_10_5_10(){
    int stateNum = 10; // 10, 15
    int actionNum = 5;
    int branchNum = 10;
    double outerStepSize = 0.05;
    double innerStepSize = 0.1;
    int maxIter = 200;
    int randomSeedNum = 2;
    int maxIterAll = 2000;
    maxIter = 100000; //large enough here, we will stop by maxIterAll
    drpgAlgGarnet(stateNum, actionNum, branchNum, outerStepSize, innerStepSize, maxIter, randomSeedNum, maxIterAll);
    double stepPolicy = 0.05; // 0.0003;
    double stepP = 0.1; // 0.00003;
    maxIter = maxIterAll / 2;

    double r1 = 10.0;
    double r2 = 10.0;
    double beta = 0.1;
    double mu = 0.1;
    dsgdaAlgGarnet(stateNum, actionNum, branchNum, stepPolicy, stepP, maxIter, r1, r2, beta, mu, randomSeedNum);
}

