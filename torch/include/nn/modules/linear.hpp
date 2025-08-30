#pragma once
#include "../module.hpp"
#include "../../../../aten/include/tensor.hpp"

namespace torch::nn {

class Linear: public Module {

public:
    Linear() = default;
    Linear(size_t in_feature, size_t out_feature, bool use_bias = true);
    virtual ~Linear() = default;
    std::shared_ptr<at::Tensor> forward(const std::shared_ptr<at::Tensor> input) override;

private:
    size_t in_feature, out_feature;
    bool use_bias;
    std::shared_ptr<at::Tensor> weight;
    std::shared_ptr<at::Tensor> bias;
};

};