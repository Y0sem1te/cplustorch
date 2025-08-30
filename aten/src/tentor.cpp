#include "../include/tensor.hpp"
#include "../include/native/ops.hpp"
#include "../../torch/include/autograd/engine.hpp"


namespace at{

Tensor::Tensor(std::vector<double> _data, std::vector<size_t> _shape, bool _require_grad) :data(_data), shape(_shape), require_grad(_require_grad){
    stride.resize(shape.size());
    size_t s = 1;
    for(i64 i=shape.size()-1;i>=0;i--){
        stride[i] = s;
        s *= shape[i];
    }
    grad.resize(data.size());
    std::fill(grad.begin(), grad.end(), 0.0);
    prev = {};
    backward_it = [](){};
}

double& Tensor::operator()(const std::vector<size_t>& idx){
    size_t s = 0;
    for(size_t i = 0; i < stride.size(); i ++ ){
        if(idx[i] >= shape[i]) throw std::out_of_range("Index out of range.");
        s += idx[i] * stride[i];
    }
    return data[s];
}

double Tensor::operator()(const std::vector<size_t>& idx) const {
    size_t s = 0;
    for(size_t i = 0; i < shape.size(); i ++ ){
        if(idx[i] >= shape[i]) throw std::out_of_range("Index out of range.");
        s += stride[i] * idx[i];
    }
    return data[s];
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other){
    return native::add(shared_from_this(), other);
};
std::shared_ptr<Tensor> Tensor::operator-(std::shared_ptr<Tensor> other){
    return native::minus(shared_from_this(), other);
};
std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other){
    return native::multiply(shared_from_this(), other);
};
std::shared_ptr<Tensor> Tensor::operator/(std::shared_ptr<Tensor> other){
    return native::divide(shared_from_this(), other);
};
std::shared_ptr<Tensor> Tensor::operator-(){
    return native::negative(shared_from_this());
};
std::shared_ptr<Tensor> Tensor::pow(double v){
    return native::pow(shared_from_this(), v);
};
std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other){
    return native::matmul(shared_from_this(), other);
};
std::shared_ptr<Tensor> Tensor::transpose(size_t dim0, size_t dim1){
    return native::transpose(shared_from_this(), dim0, dim1);
}

void Tensor::backward(){
    torch::autograd::Engine eng(shared_from_this());
    eng.topsort();
    eng.execute();
}

}; // namespace at