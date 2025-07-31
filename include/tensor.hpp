/*Tensor class*/

#pragma once
#include <vector>
#include <iostream>
#include <set>
#include <memory>
#include <functional>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    /*tensor type*/
    std::vector<double> data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    std::vector<double> grad;
    std::function<void()> backward;
    std::string op;
    std::set<std::shared_ptr<Tensor>> prev;
    bool requires_grad = true;

    /*functions*/

    /*constructor*/
    Tensor(const std::vector<double>& data_, 
           const std::vector<size_t>& shape_,
           const std::string &op = "",
           bool requires_grad_ = true);

    /*index in flattened tensor*/
    size_t idx(const std::vector<size_t>& indices) const;

    /*indexing*/
    double& operator()(const std::initializer_list<size_t>& indices);
    const double& operator()(const std::initializer_list<size_t>& indices) const;

    /* calculate between tensor */
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);  //add
    std::shared_ptr<Tensor> operator-();                               //negate
    std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> other);  //minus
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);  //multiply
    std::shared_ptr<Tensor> pow(const double other);                   //power
    std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> other);  //divide
    std::shared_ptr<Tensor> matmul2d(std::shared_ptr<Tensor> other);   //matrix multiply
    std::shared_ptr<Tensor> transpose(size_t a, size_t b);             //transpose

    /* broadcasting */
    std::vector<size_t> broadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) const;
    std::vector<size_t> broadcastIdx(const std::vector<size_t>& idx, const std::vector<size_t>& org_shape) const;
    void traverse(const std::vector<size_t>& shape, std::function<void(const std::vector<size_t>&)> func) const;

    /* backward function */
    const std::vector<std::shared_ptr<Tensor>> topSort();
    void backwardAll();
};