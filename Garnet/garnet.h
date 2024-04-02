#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <assert.h>
#include <chrono>
#include <limits>
#include <cmath>
#include <memory>
#include <functional>
#include <sys/timeb.h>
#include <ctime>
#include <fstream>

using namespace std;


/**
 * Returns sorted indexes for the given array (in ascending order).
 */
template <typename T>
inline vector<size_t> sort_indexes_ascending(vector<T> const& v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    // initialize to index positions
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}
template <typename T>
inline vector<size_t> sort_indexes(std::vector<T> const& v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

/**
 * Returns sorted indexes for the given array (in descending order).
 */
template <typename T>
inline vector<size_t> sort_indexes_descending(vector<T> const& v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    // initialize to index positions
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
    return idx;
}


/** This is useful functionality for debugging.  */
template<class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (const auto& p : vec) {
        cout << p << " ";
    }
    return os;
}

/*
 This is a CLASS that contains all the information for problem instance
 */
class MDP {
public:
    // 3D vector
    vector<vector<vector<double>>> P;       // transition kernel 
    vector<vector<vector<double>>> B;       // Branch location kernel
    vector<vector<vector<double>>> cost;    // cost function
    double gamma;                           // discount factor
    vector<double> rho;                     // initial distribution

    // auxiliary MDPs variables
    size_t nActions;                           // get the number of actions
    size_t nStates;                            // get the number of states

    // create auxiliary variables from input variables
    void createAuxiliaryVar() {
        // define nAction and nStates, and initialize pb_max and b_max_min
        nStates = P.size();
        nActions = P[0].size();
    }
    
    // view the transition kernel etc;
    void view(){
        for (int s = 0; s < nStates; ++s){
            for (int a = 0; a < nActions; ++a){
                for (int sPlus = 0; sPlus < nStates; ++sPlus) {
                    std::cout<<"P(" << s << "," << a << "," << sPlus <<")" << P[s][a][sPlus]<<"\n";
                }
            }
        }

        for (int s = 0; s < nStates; ++s){
            for (int a = 0; a < nActions; ++a){
                for (int sPlus = 0; sPlus < nStates; ++sPlus) {
                    std::cout<<"B(" << s << "," << a << "," << sPlus <<")" << B[s][a][sPlus]<<"\n";
                }
            }
        }
        
        for (int s = 0; s < nStates; ++s){
            for (int a = 0; a < nActions; ++a){
                for (int sPlus = 0; sPlus < nStates; ++sPlus) {
                    std::cout<<"cost(" << s << "," << a << "," << sPlus <<")" << cost[s][a][sPlus]<<"\n";
                }
            }
        }
        
        std::cout << "gamma:" << gamma << "\n";
        for (int s = 0; s < nStates; ++s){
            std::cout<<"rho(" << s <<")" << rho[s]<<"\n";
        }
    }
};

extern double inner_product(const vector<double>& v1, const vector<double>& v2);
double PHIVal(
    const MDP& mdpInstance, 
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P_0,
    const vector<vector<double>>& kappa,
    const double& stepsize,
    const int& max_iter
);

vector<double> mdp_value(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P
);

vector<int> P_branch(size_t nStates, size_t nBranchs, int seed);
vector<vector<double>> random_policy(const MDP& mdpInstance, int seed);
vector<vector<vector<double>>> dsgda_grad_P(
    const MDP& mdpInstance,
    const vector<vector<double>> &policy,
    const vector<vector<vector<double>>> &P,
    const vector<vector<vector<double>>> &Pbar,
    const double r2,
    const vector<double>& eta,
    const vector<double>& v_now
);


vector<vector<double>> dsgda_grad_policy(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<double>>& policyBar,
    const double r1,
    const vector<vector<vector<double>>>& P,
    const vector<double>& eta,
    const vector<double>& v_now
);


vector<vector<double>> mdp_grad_policy(
    const MDP& mdpInstance,
    const vector<vector<double>>& policy,
    const vector<vector<vector<double>>>& P,
    const vector<double>& eta,
    const vector<double>& v_now
);

vector<vector<vector<double>>> MDP_grad_P(
    const MDP& mdpInstance,
    const vector<vector<double>> &policy,
    const vector<vector<vector<double>>> &P,
    const vector<double>& eta,
    const vector<double>& v_now
);

vector<double> mdp_occupancy(
    const MDP& mdpInstance,
    const vector<vector<vector<double>>>& PNow,
    const vector<vector<double>>& policyNow
);

double L2_Gap(const vector<double>& v1, const vector<double>& v2);

void garnetSA_5_6_3();
void garnetSA_10_5_10();
void garnet_5_6_3();
void garnet_10_5_10();
