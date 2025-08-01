#pragma once
#include "../tensor.hpp"

class MSELoss{
public:
    MSELoss();
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> label);
};