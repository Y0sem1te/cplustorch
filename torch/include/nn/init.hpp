#pragma once

#include <cmath>
#include <random>
#include <memory>
#include <vector>
#include "../../../aten/include/tensor.hpp"

namespace torch::nn::init {
    void kaimingUniform(at::Tensor& tensor);
}