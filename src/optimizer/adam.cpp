#include "../../include/optimizer/adam.hpp"
#include <cmath>

Adam::Adam(
    std::shared_ptr<Tensor> _parameters,
    double _learning_rate,
    double _beta1,
    double _beta2,
    double _eps,
    double _weight_decay
):parameters(_parameters), learning_rate(_learning_rate), beta1(_beta1), beta2(_beta2), eps(_eps), weight_decay(_weight_decay)
{
    time = 0;
    m = std::make_shared<Tensor>(std::vector<double>(parameters->data.size(), 0.0), parameters->shape);
    v = std::make_shared<Tensor>(std::vector<double>(parameters->data.size(), 0.0), parameters->shape);
}

void Adam::zero_grad() {
    parameters->grad = std::vector<double>(parameters->data.size(), 0.0);
}

void Adam::step() {
    time++;
    for (size_t i = 0; i < parameters->data.size(); ++i)
        m->data[i] = beta1 * m->data[i] + (1 - beta1) * parameters->grad[i];
    for (size_t i = 0; i < parameters->data.size(); ++i)
        v->data[i] = beta2 * v->data[i] + (1 - beta2) * parameters->grad[i] * parameters->grad[i];

    auto m_hat = std::make_shared<Tensor>(m->data, m->shape);
    auto v_hat = std::make_shared<Tensor>(v->data, v->shape);
    for (size_t i = 0; i < m_hat->data.size(); ++i) {
        m_hat->data[i] /= (1 - std::pow(beta1, time));
        v_hat->data[i] /= (1 - std::pow(beta2, time));
    }
    for (size_t i = 0; i < parameters->data.size(); ++i) {
        parameters->data[i] -= learning_rate * (m_hat->data[i] / (std::sqrt(v_hat->data[i]) + eps));
        if (weight_decay > 0.0)
            parameters->data[i] -= learning_rate * weight_decay * parameters->data[i];
    }
}