#include "../../include/nn/linear.hpp"

Linear::Linear(size_t input_dim, size_t output_dim, bool use_bias)
 :in_features(input_dim), out_features(output_dim), use_bias(use_bias) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);
    std::vector<double> result(input_dim*output_dim, 0.0);
    weights = std::make_shared<Tensor>(result, std::vector<size_t>{output_dim, input_dim});
    for(size_t i = 0; i < input_dim * output_dim; ++i) {
        weights->data[i] = dist(gen);
    }
    if (use_bias) {
        std::vector<double> data(output_dim, 0.0);
        bias = std::make_shared<Tensor>(data, std::vector<size_t>{1, output_dim});
    }else {
        bias = nullptr;
    }
}

std::shared_ptr<Tensor> Linear::operator()(std::shared_ptr<Tensor> input){
    if(input->shape.back() != weights->shape[1]) throw std::out_of_range("Indices dimension mismatch.");
    if(input->shape.size() == 1) input->shape.assign({1, input->shape[0]});
    if(input->shape.size() != 2){
        size_t front_value = 1;
        for(size_t i=0; i<input->shape.size()-1;i++){
            front_value *= input->shape[i];
        }
        input->shape.assign({front_value, input->shape.back()});
    }
    std::shared_ptr<Tensor> res = input->matmul2d(weights->transpose(0, 1));
    if(use_bias) res = *res + bias;
    return res;
}