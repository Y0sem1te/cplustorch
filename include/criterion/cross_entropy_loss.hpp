#pragma
#include "../tensor.hpp"

class CrossEntropyLoss{
public:
    CrossEntropyLoss();
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> label);
};