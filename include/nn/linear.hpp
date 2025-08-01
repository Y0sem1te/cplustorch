#pragma once
#include "../tensor.hpp"
#include <random>
#include <memory>
class Linear{
private:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    size_t in_features;
    size_t out_features;
    bool use_bias;

public:
    Linear(size_t input_dim, size_t output_dim, bool use_bias = true);
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input);
};