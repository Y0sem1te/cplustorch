#include "../../include/criterion/cross_entropy_loss.hpp"
#include <cmath>

CrossEntropyLoss::CrossEntropyLoss() {};

std::shared_ptr<Tensor> CrossEntropyLoss::operator()(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> label) {
    if (output->data.size() != label->data.size()) {
        throw std::invalid_argument("Output and label tensors must have the same shape.");
    }

    double loss = 0.0;
    for (size_t i = 0; i < output->data.size(); ++i) {
        loss -= label->data[i] * std::log(output->data[i] + 1e-15); // Adding a small constant to avoid log(0)
    }
    
    auto result = std::make_shared<Tensor>(std::vector<double>{loss}, std::vector<size_t>{1}, "CrossEntropyLoss");
    result->prev.insert(output);
    result->prev.insert(label);
    result->backward = [output, label, result]() {
        for (size_t i = 0; i < output->data.size(); ++i) {
            double grad = -label->data[i] / (output->data[i] + 1e-15);
            output->grad[i] += grad * result->grad[0];
            label->grad[i] += -std::log(output->data[i] + 1e-15) * result->grad[0];
        }
    };
    return result;
}
