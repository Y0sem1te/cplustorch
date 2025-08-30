#pragma once
#include "../module.hpp"
#include "../../../../aten/include/tensor.hpp"
#include <memory>

namespace torch::nn {

class MSELoss : public Module {
public:
    MSELoss() = default;
    virtual ~MSELoss() = default;
    std::shared_ptr<at::Tensor> forward(const std::shared_ptr<at::Tensor> input) override;
    std::shared_ptr<at::Tensor> forward(const std::shared_ptr<at::Tensor> prediction, const std::shared_ptr<at::Tensor> target);

private:
    double mean(const std::vector<double>& data);
};

} // namespace torch::nn