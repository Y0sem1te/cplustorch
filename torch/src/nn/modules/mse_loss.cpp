#include "../../../include/nn/modules/mse_loss.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace torch::nn {

std::shared_ptr<at::Tensor> MSELoss::forward(const std::shared_ptr<at::Tensor> input) {
    throw std::runtime_error("MSELoss requires both prediction and target. Use forward(prediction, target) instead.");
}

std::shared_ptr<at::Tensor> MSELoss::forward(const std::shared_ptr<at::Tensor> prediction, 
                                           const std::shared_ptr<at::Tensor> target) {
    if (!prediction || !target) throw std::invalid_argument("Prediction and target cannot be null");
    if (prediction->shape != target->shape) throw std::invalid_argument("Prediction and target must have the same shape");
    if (prediction->data.size() != target->data.size()) throw std::invalid_argument("Prediction and target must have the same number of elements");

    std::vector<double> diff_squared;
    diff_squared.reserve(prediction->data.size());
    for (size_t i = 0; i < prediction->data.size(); ++i) {
        double diff = prediction->data[i] - target->data[i];
        diff_squared.push_back(diff * diff);
    }
    
    double mse_value = mean(diff_squared);
    std::shared_ptr<at::Tensor> result = std::make_shared<at::Tensor>(
        std::vector<double>{mse_value}, 
        std::vector<size_t>{1}, 
        prediction->require_grad || target->require_grad
    );

    if (result->require_grad) {
        result->prev.insert(prediction);
        result->prev.insert(target);
        
        result->backward_it = [prediction, target, result]() {
            if (prediction->require_grad) {
                double n = static_cast<double>(prediction->data.size());
                prediction->grad.resize(prediction->data.size());
                for (size_t i = 0; i < prediction->data.size(); ++i) {
                    double grad = 2.0 * (prediction->data[i] - target->data[i]) / n;
                    prediction->grad[i] += grad * result->grad[0];
                }
            }
            
            if (target->require_grad) {
                double n = static_cast<double>(target->data.size());
                target->grad.resize(target->data.size());
                for (size_t i = 0; i < target->data.size(); ++i) {
                    double grad = -2.0 * (prediction->data[i] - target->data[i]) / n;
                    target->grad[i] += grad * result->grad[0];
                }
            }
        };
    }
    
    return result;
}

double MSELoss::mean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>(data.size());
}

} // namespace torch::nn