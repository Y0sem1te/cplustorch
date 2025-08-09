#pragma once
#include <memory>
#include <vector>
#include <set>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace at {

class Tensor : public std::enable_shared_from_this<Tensor> {

public:
    std::vector<double> data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    std::set<std::shared_ptr<Tensor>> prev;
    std::vector<double> grad;
    std::function<void()>backward_it;
    bool require_grad;

    using i64 = long long;
    Tensor(std::vector<double> data, std::vector<size_t> shape, bool require_grad = false);
    Tensor() = default;
    Tensor(const Tensor &org) = default;
    Tensor& operator=(const Tensor& org) = default;
    double& operator()(const std::vector<size_t>& idx);
    double operator()(const std::vector<size_t>& idx) const;
    virtual ~Tensor() = default;

    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator-();
    std::shared_ptr<Tensor> pow(double v);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> transpose(size_t dim0, size_t dim1);

    void backward();
};

} // namespace at