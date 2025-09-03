#include "../../include/nn/init.hpp"

namespace torch::nn::init{
    void kaimingUniform(at::Tensor& tensor){
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // For a 2D weight tensor with shape [out_features, in_features]
        // fan_in is the number of input features (tensor.shape[1])
        double fan_in = tensor.shape.size() >= 2 ? tensor.shape[1] : tensor.shape[0];
        double bound = std::sqrt(6.0 / fan_in);
        std::uniform_real_distribution<> dis(-bound, bound);

        for (auto& elem : tensor.data) {
            elem = dis(gen);
        }
    }
};