#pragma once
#include "../tensor.hpp"

class Adam {
private:
    std::shared_ptr<Tensor> parameters;
    double learning_rate;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    std::shared_ptr<Tensor> m; // First moment vector
    std::shared_ptr<Tensor> v; // Second moment vector
    size_t time; // Time step

public:
    Adam(
        std::shared_ptr<Tensor> parameters,
        double learning_rate = 0.001,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-8,
        double weight_decay = 0.0
    );
    void zero_grad();
    void step();
};