#include "../../include/nn/init.hpp"

namespace torch::nn::init{
    void kaimingUniform(at::Tensor& tensor){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-std::sqrt(6.0 / tensor.shape.back()), std::sqrt(6.0 / tensor.shape.back()));

        for (auto& elem : tensor.data) {
            elem = dis(gen);
        }
    }
};