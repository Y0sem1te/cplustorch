/*
    activation functions
    relu, sigmoid, tanh
*/

#include "../include/functional.hpp"
#include <cmath>
namespace Functional
{
    std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x)
    {
        std::vector<double> res(x->data.size());
        for (size_t i = 0; i < x->data.size(); i++)
            res[i] = std::max(0.0, x->data[i]);
        auto out = std::make_shared<Tensor>(res, x->shape, "relu");
        auto self = x->shared_from_this();
        out->prev.insert(self);
        out->backward = [self, out](){
            if(self->requires_grad == false) return;
            for (size_t i = 0; i < self->data.size(); i++){
                double self_val = self->data[i];
                self->grad[i] += (self_val > 0 ? 1 : 0) * out->grad[i];
            }
        };
        return out;
    }

    std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> x){
        std::vector<double> res(x->data.size());
        for(size_t i=0; i < x->data.size(); i ++){
            res[i] = 1.0 / (1.0 + std::exp(-x->data[i]));
        }
        auto out = std::make_shared<Tensor>(res, x->shape, "sigmoid");
        out->prev.insert(x);
        out->backward = [x, out](){
            if(x -> requires_grad == false) return;
            for(size_t i = 0; i < x -> grad.size(); i ++ ){
                x->grad[i] += out->data[i] * (1 - out->data[i]) * out->grad[i];
            }
        };
        return out;
    }

    std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> x){
        std::vector<double> res(x->data.size());
        for(size_t i = 0; i < x->data.size(); i++)
            res[i] = std::tanh(x->data[i]);
        auto out = std::make_shared<Tensor>(res, x->shape, "tanh");
        out->prev.insert(x);
        out->backward = [x, out](){
            if(x->requires_grad == false) return;
            for(size_t i = 0; i < x->grad.size(); i ++ ){
                x->grad[i] += (1 - std::tanh(x->data[i])*std::tanh(x->data[i])) * out->grad[i];
            }
        };
        return out;
    }
}
