#pragma once
#include <numeric>
#include <stdexcept>
#include <memory>
#include <vector>

namespace at {

class Tensor;

namespace native {

    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> minus(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> divide(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> negative(std::shared_ptr<Tensor> self);
    std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> self, double v);
    std::shared_ptr<Tensor> matmul2d(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> self, size_t dim0, size_t dim1);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other);

} // namespace native

} //namespace at