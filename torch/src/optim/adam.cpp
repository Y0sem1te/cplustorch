#include "../../include/optim/adam.hpp"
#include "../../../aten/include/tensor.hpp"
#include <cmath>

namespace torch::optim {

Adam::Adam(const std::vector<std::shared_ptr<at::Tensor>>& params,
           double lr,
           double beta1,
           double beta2,
           double eps)
    : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    states_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        auto p = params_[i];
        size_t n = p->data.size();
        states_[i].m.assign(n, 0.0);
        states_[i].v.assign(n, 0.0);
    }
}

void Adam::step() {
    step_ += 1;
    
    // Gradient clipping to prevent explosion
    double max_grad_norm = 1.0;
    double total_norm = 0.0;
    
    // Calculate total gradient norm
    for (size_t i = 0; i < params_.size(); ++i) {
        auto &grad = params_[i]->grad;
        if (grad.size() == 0) continue;
        
        for (size_t j = 0; j < grad.size(); ++j) {
            total_norm += grad[j] * grad[j];
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // Clip gradients if needed
    double clip_coef = max_grad_norm / (total_norm + 1e-6);
    if (clip_coef < 1.0) {
        for (size_t i = 0; i < params_.size(); ++i) {
            auto &grad = params_[i]->grad;
            for (size_t j = 0; j < grad.size(); ++j) {
                grad[j] *= clip_coef;
            }
        }
    }
    
    // Adam optimization step
    for (size_t i = 0; i < params_.size(); ++i) {
        auto p = params_[i];
        auto &grad = p->grad;
        auto &data = p->data;
        auto &m = states_[i].m;
        auto &v = states_[i].v;

        if (grad.size() == 0) continue;

        size_t n = grad.size();
        for (size_t j = 0; j < n; ++j) {
            double g = grad[j];
            m[j] = beta1_ * m[j] + (1.0 - beta1_) * g;
            v[j] = beta2_ * v[j] + (1.0 - beta2_) * (g * g);

            double m_hat = m[j] / (1.0 - std::pow(beta1_, step_));
            double v_hat = v[j] / (1.0 - std::pow(beta2_, step_));

            double update = lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            data[j] -= update;
        }
    }
}

void Adam::zero_grad() {
    for (auto &param : params_) {
        std::fill(param->grad.begin(), param->grad.end(), 0.0);
    }
}

} // namespace torch::optim