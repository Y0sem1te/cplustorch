#include "./aten/include/tensor.hpp"
#include "./aten/include/native/ops.hpp"
#include "./torch/include/autograd/engine.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

using namespace at;

bool isClose(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

void printTensor(std::shared_ptr<Tensor> tensor, const std::string& name) {
    std::cout << name << " values: [";
    for(size_t i = 0; i < tensor->data.size(); i++) {
        std::cout << std::fixed << std::setprecision(3) << tensor->data[i];
        if(i < tensor->data.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << name << " grads:  [";
    for(size_t i = 0; i < tensor->grad.size(); i++) {
        std::cout << std::fixed << std::setprecision(3) << tensor->grad[i];
        if(i < tensor->grad.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<double>{4.0, 5.0}, std::vector<size_t>{2}, true);
    std::shared_ptr<Tensor> c = std::make_shared<Tensor>(std::vector<double>{5.0, 7.0}, std::vector<size_t>{2}, true);
    std::shared_ptr<Tensor> d = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0, 3.0, 5.0, 6.0, 7.0}, std::vector<size_t>{3, 2}, true);
    std::shared_ptr<Tensor> o  = (*(*(*(*a - b)).pow(3.0) / c)).matmul((*d).transpose(0, 1));
    o->backward();
    printTensor(o, "o");
    // Check if the gradients are correct
    printTensor(d, "d");
    printTensor(c, "c");
    printTensor(a, "a");
    printTensor(b, "b");
    return 0;
}
