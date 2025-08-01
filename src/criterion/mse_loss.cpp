#include "../../include/criterion/mse_loss.hpp"
#include <cmath>

MSELoss::MSELoss() {};

std::shared_ptr<Tensor> MSELoss::operator()(std::shared_ptr<Tensor>output, std::shared_ptr<Tensor>label){
    if(output->data.size() != label->data.size()) throw std::out_of_range("Dimensions mismatch.");
    double loss = 0.0;
    size_t n = output->data.size();
    for(size_t i=0;i<output->data.size();i++){
        loss += std::pow(output->data[i]-label->data[i], 2);
    }
    loss /= n;
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(std::vector<double>{loss}, std::vector<size_t>{1}, "MSELoss");
    result->prev.insert(output);
    result->prev.insert(label);
    result->backward = [output, label, result](){
        for(size_t i=0;i<output->grad.size();i++){
            output->grad[i] += result->grad[0] * 2 * (output->data[i] - label->data[i]) / output->data.size();
            label->grad[i] += -result->grad[0] * 2 * (output->data[i] - label->data[i]) / output->data.size();
        }
    };
    return result;
}
