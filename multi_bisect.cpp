#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

extern "C" {

typedef struct {
    int size;
    double* values;
} DataPoint;

typedef struct {
    int size;
    DataPoint* points;
} Data;

Data* test(Data* data) {
    for (int i=0; i < data->size; ++i){
        for (int j=0; j < data->points[i].size; ++j){
            std::cout << data->points[i].values[j]<< std::endl;
        }
    }
    return data;
}

// Function to find the top k elements based on a given lambda
Data* multi_bisect_v2(
    Data* data, int n, int k, double* constraints, int constraints_size,
    double lb, double ub, double epsilon, int max_loop, bool verbose
) {
    // timer
    auto timer_start = std::chrono::high_resolution_clock::now();

    // Convert C-style data to C++ vector for processing
    std::vector<std::vector<double>> cpp_data;
    for (int j = 0; j < n; ++j) {
        std::vector<double> point(data->points[j].values, data->points[j].values + data->points[j].size);
        point.push_back(point[0]);  // 排序分
        cpp_data.push_back(point);
    }

    if (verbose) {
        for (double lambda : cpp_data[0]) {
            std::cout << lambda << " ";
        }
        std::cout << std::endl;
    }

    std::vector<double> lambdas(constraints_size + 1, 0);
    lambdas[0] = 1;
    int iter = 0;
    int loop_times = 0;
    double best_opt = 0;
    std::vector<double> best_lambdas(constraints_size + 1, 0);
    int best_loop = 0;
    int best_iter = 0;

    std::vector<bool> constraints_satisfied(constraints_size, true);

    auto compare = [](const std::vector<double>& a, const std::vector<double>& b) {return a.back() > b.back();};

    // 初始化结果集,提前分配内存
    Data* result = new Data{
        static_cast<int>(k), // Data的大小是外层vector的大小
        new DataPoint[k]     // 为每个DataPoint分配内存
    };
    for (size_t i = 0; i < k; ++i) {
        result->points[i].size = static_cast<int>(constraints_size + 1);
        result->points[i].values = new double[constraints_size + 1];
    }

    while (loop_times < max_loop) {
        loop_times++;
        for (size_t i = 1; i < constraints_size + 1; i++) {
            double l = lb, r = ub;
            while (r >= l + epsilon) {
                iter++;
                double lambdas_i = (r + l) / 2;
                double lambdas_i_delta = lambdas_i - lambdas[i];

                double lambdas_0 = std::max(0.0, lambdas[0] - lambdas_i_delta);
                double lambdas_0_delta = lambdas_0 - lambdas[0];
                if (verbose) {
                    std::cout << "lambdas update: " << i << " " << lambdas_i << " " << lambdas[i] << " " << lambdas_0 << " " << lambdas[0] << std::endl; 
                }

                // update ranking score
                for (size_t j = 0; j < cpp_data.size(); j++) {
                    cpp_data[j].back() += lambdas_0_delta * cpp_data[j][0];
                    cpp_data[j].back() += lambdas_i_delta * cpp_data[j][i];
                }
                lambdas[0] = lambdas_0;
                lambdas[i] = lambdas_i;

                // Find top k elements
                std::nth_element(cpp_data.begin(), cpp_data.begin() + k - 1, cpp_data.end(), compare);

                // Check constraints
                bool all_constraints_satisfied = true;
                for (int ci = 0; ci < constraints_size; ci++) {
                    double sum = 0;
                    for (size_t j = 0; j < k; j++){
                        sum += cpp_data[j][ci+1];
                    }
                    constraints_satisfied[ci] = sum >= constraints[ci];
                    if (!constraints_satisfied[ci]) {
                        all_constraints_satisfied = false;
                    }
                }

                if (verbose) {
                    std::cout << "lambdas: [";
                    for (double lambda: lambdas) {
                        std::cout << lambda << ", ";
                    }
                    std::cout << "] ";
                    for (int ci = 0; ci < constraints_size; ci++) {
                        double sum = 0;
                        for (size_t j = 0; j < k; j++){
                            sum += cpp_data[j][ci+1];
                        }
                        std::cout << "c_" << ci << ": " << constraints[ci] << "|" << sum;
                    }
                    std::cout << std::endl;
                }

                if (all_constraints_satisfied) {
                    double opt = 0;
                    for (size_t j = 0; j < k; j++){
                        opt += cpp_data[j][0];
                    }
                    if (opt > best_opt) {
                        if (verbose) {
                            std::cout << "better solution found! opt: " << opt << "lambdas: ";
                            for (double lambda: lambdas) {
                                std::cout << lambda << ", ";
                            }
                            std::cout << "] " << std::endl;
                        }
                        best_lambdas = lambdas;
                        best_opt = opt;
                        best_loop = loop_times;
                        best_iter = iter;
                        for (size_t j = 0; j < k; j++) {
                            std::copy(cpp_data[j].begin(), cpp_data[j].begin() + constraints_size + 1, result->points[j].values);
                        }
                    }
                }

                if (constraints_satisfied[i-1]) {
                    r = lambdas[i];
                } else {
                    l = lambdas[i];
                }
            }
        }
    }

    if (verbose) {
        std::cout << "loop: " << best_loop << " iter: " << best_iter << " opt: " << best_opt << " lambdas: [";
        for (double lambda : best_lambdas) {
            std::cout << lambda << ",";
        }
        std::cout << "]" << std::endl;
    }

    auto timer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = timer_end - timer_start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    
    return result;
}

} // extern "C"
