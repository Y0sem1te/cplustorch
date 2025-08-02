/*
    activation functions
*/
#pragma once
#include <memory>
#include <vector>
#include "../include/tensor.hpp"
namespace Functional {
    std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> x);
}