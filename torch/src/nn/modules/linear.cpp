#include "../../../include/nn/module.hpp"
#include "../../../include/nn/modules/linear.hpp"
#include "../../../include/nn/init.hpp"

namespace torch::nn {
    
    Linear::Linear(size_t in_feature, size_t out_feature, bool use_bias)
        : Module(), in_feature(in_feature), out_feature(out_feature), use_bias(use_bias) {
        std::vector<double> wdata(in_feature*out_feature, 0.0);
        std::vector<size_t> wshape = {out_feature, in_feature};
        weight = std::make_shared<at::Tensor>(wdata, wshape, true);
        torch::nn::init::kaimingUniform(*weight);
        register_parameter("w", weight);
        if (use_bias) {
            std::vector<double> bdata(out_feature, 0.0);
            std::vector<size_t> bshape = {out_feature};
            bias = std::make_shared<at::Tensor>(bdata, bshape, true);
            register_parameter("b", bias);
        }
    }

    std::shared_ptr<at::Tensor> Linear::forward(const std::shared_ptr<at::Tensor> input) {
        // input @ weight.T + bias
        auto result = input->matmul(weight->transpose(0, 1));
        if (use_bias) {
            result = (*result) + (bias);
        }
        return result;
    }

};