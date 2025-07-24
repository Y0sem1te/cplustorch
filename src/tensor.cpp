#include "../include/tensor.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
/**
 * @brief Construct a new Tensor object
 */
Tensor::Tensor(const std::vector<double>& data_, 
               const std::vector<size_t>& shape_,
               const std::string &op) 
            :data(data_), shape(shape_), op(op){
    grad.resize(data.size());
    std::fill(grad.begin(), grad.end(), 0.0);
    stride.resize(shape.size());
    size_t s = 1;
    for(int i = (int)shape.size()-1; i >= 0; i --){
        stride[i] = s;
        s *= shape[i];
    }
    backward = [](){};
}
// transfer
size_t Tensor::idx(const std::vector<size_t>& indices) const {
    if(indices.size() != shape.size())
        throw std::out_of_range("Indices dimension mismatch");
    size_t offset = 0;
    for(size_t i=0;i<indices.size();i++){
        if(indices[i] >= shape[i])
            throw std::out_of_range("Index out of bounds");
        offset += indices[i] * stride[i];
    }
    return offset;
}

// indexing
double& Tensor::operator()(const std::initializer_list<size_t>& indices) {
    return data[idx(std::vector<size_t>(indices))];
}
const double& Tensor::operator()(const std::initializer_list<size_t>& indices) const {
    return data[idx(std::vector<size_t>(indices))];
}

/* arithmetic operations */
/*add*/
std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other){
    std::vector<size_t> out_shape = broadcastShape(shape, other->shape);
    size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<size_t>());
    std::vector<double> res(out_size);
    traverse(out_shape, [&](const std::vector<size_t>& out_idx){
        size_t flatten_idx = 0, stride_val = 1;
        for(int i = (int)out_shape.size() - 1; i >=0 ; i--) {
            flatten_idx += out_idx[i] * stride_val;
            stride_val *= out_shape[i];
        }
        std::vector<size_t> self_idx = broadcastIdx(out_idx, shape);
        std::vector<size_t> other_idx = broadcastIdx(out_idx, other->shape);
        double self_val = data[idx(self_idx)], other_val = other->data[other->idx(other_idx)];
        res[flatten_idx] = self_val + other_val;
    });
    auto out = std::make_shared<Tensor>(res, out_shape, "+");
    auto self = shared_from_this();
    out->prev.insert(self);
    out->prev.insert(other);
    out->backward = [out_shape, self, other, out](){
        out->traverse(out_shape, [&](const std::vector<size_t>& out_idx){
            size_t flatten_idx = 0, stride_val = 1;
            for(int i = (int)out_shape.size() - 1; i >= 0; i--){
                flatten_idx += out_idx[i] * stride_val;
                stride_val *= out_shape[i];
            }
            std::vector<size_t> self_idx = self->broadcastIdx(out_idx, self->shape);
            std::vector<size_t> other_idx = other->broadcastIdx(out_idx, other->shape);
            double self_val = self->data[self->idx(self_idx)], other_val = other->data[other->idx(other_idx)];
            self->grad[self->idx(self_idx)] += 1.0 * out->grad[flatten_idx];
            other->grad[other->idx(other_idx)] += 1.0 * out->grad[flatten_idx];
        });
    };
    return out;
}
/*negative*/
std::shared_ptr<Tensor> Tensor::operator-() {
    std::vector<double> res(data.size());
    for(size_t i = 0; i < data.size(); i++)
        res[i] = -data[i];
    auto out = std::make_shared<Tensor>(res, shape, "neg");
    auto self = shared_from_this();
    out->prev.insert(self);
    out->backward = [self, out](){
        for(size_t i = 0; i < self->data.size(); i++)
            self->grad[i] = -1.0 * out->grad[i];
    };
    return out;
}
/*minus*/
std::shared_ptr<Tensor> Tensor::operator-(std::shared_ptr<Tensor> other) {
    std::vector<size_t> out_shape = broadcastShape(shape, other->shape);
    size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<size_t>());
    std::vector<double> res(out_size);
    traverse(out_shape, [&](const std::vector<size_t>& out_idx){
        size_t flatten_idx = 0, stride_val = 1;
        for(int i = (int)out_shape.size() - 1; i >=0 ; i--) {
            flatten_idx += out_idx[i] * stride_val;
            stride_val *= out_shape[i];
        }
        std::vector<size_t> self_idx = broadcastIdx(out_idx, shape);
        std::vector<size_t> other_idx = broadcastIdx(out_idx, other->shape);
        double self_val = data[idx(self_idx)], other_val = other->data[other->idx(other_idx)];
        res[flatten_idx] = self_val - other_val;
    });
    auto out = std::make_shared<Tensor>(res, out_shape, "-");
    auto self = shared_from_this();
    out->prev.insert(self);
    out->prev.insert(other);
    out->backward = [out_shape, self, other, out](){
        out->traverse(out_shape, [&](const std::vector<size_t>& out_idx){
            size_t flatten_idx = 0, stride_val = 1;
            for(int i = (int)out_shape.size() - 1; i >= 0; i--){
                flatten_idx += out_idx[i] * stride_val;
                stride_val *= out_shape[i];
            }
            std::vector<size_t> self_idx = self->broadcastIdx(out_idx, self->shape);
            std::vector<size_t> other_idx = other->broadcastIdx(out_idx, other->shape);
            double self_val = self->data[self->idx(self_idx)], other_val = other->data[other->idx(other_idx)];
            self->grad[self->idx(self_idx)] += 1.0 * out->grad[flatten_idx];
            other->grad[other->idx(other_idx)] += -1.0 * out->grad[flatten_idx];
        });
    };
    return out;
}
/*multiply*/
std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
    std::vector<size_t> out_shape = broadcastShape(shape, other->shape);
    size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<size_t>());
    std::vector<double> res(out_size);
    traverse(out_shape, [&](const std::vector<size_t>& out_idx){
        size_t flatten_idx = 0, stride_val = 1;
        for(int i = (int)out_shape.size() - 1; i >= 0; i--) {
            flatten_idx += out_idx[i] * stride_val;
            stride_val *= out_shape[i];
        }
        std::vector<size_t> self_idx = broadcastIdx(out_idx, shape);
        std::vector<size_t> other_idx = broadcastIdx(out_idx, other->shape);
        double self_val = data[idx(self_idx)], other_val = other->data[other->idx(other_idx)];
        res[flatten_idx] = self_val * other_val;
    });
    auto out = std::make_shared<Tensor>(res, out_shape, "*");
    auto self = shared_from_this();
    out->prev.insert(self);
    out->prev.insert(other);
    out->backward = [out_shape, self, other, out](){
        out->traverse(out_shape, [&](const std::vector<size_t>& out_idx){
            size_t flatten_idx = 0, stride_val = 1;
            for(int i = (int)out_shape.size() - 1; i >= 0; i--){
                flatten_idx += out_idx[i] * stride_val;
                stride_val *= out_shape[i];
            }
            std::vector<size_t> self_idx = self->broadcastIdx(out_idx, self->shape);
            std::vector<size_t> other_idx = other->broadcastIdx(out_idx, other->shape);
            double self_val = self->data[self->idx(self_idx)], other_val = other->data[other->idx(other_idx)];
            self->grad[self->idx(self_idx)] += other_val * out->grad[flatten_idx];
            other->grad[other->idx(other_idx)] += self_val * out->grad[flatten_idx];
        });
    };
    return out;
}
/*power*/
std::shared_ptr<Tensor> Tensor::pow(const double other) {
    std::vector<double> res(data.size());
    for(size_t i = 0; i < data.size(); i++)
        res[i] = std::pow(data[i], other);
    auto out = std::make_shared<Tensor>(res, shape, "**");
    auto self = shared_from_this();
    out->prev.insert(self);
    out->backward = [self, other, out](){
       for(size_t i = 0; i < self->data.size(); i++) {
            double self_val = self->data[i];
            self->grad[i] += other * std::pow(self_val, other - 1) * out->grad[i];
        }
    };
    return out;
}
/*divide*/
std::shared_ptr<Tensor> Tensor::operator/(std::shared_ptr<Tensor> other) {
    return *this * other->pow(-1.0);
}

/* broadcasting */
/* shape */
std::vector<size_t> Tensor::broadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) const {
    size_t n = std::max(a.size(), b.size());
    std::vector<size_t>result(n);
    for(size_t i = 0; i < n; i ++ ) {
        size_t dim_a = (i < n - a.size() ? 1 : a[i - (n - a.size())]);
        size_t dim_b = (i < n - b.size() ? 1 : b[i - (n - b.size())]);
        if (dim_a != 1 && dim_b != 1 && dim_a != dim_b) {
            throw std::invalid_argument("Shapes are not broadcastable");
        }
        result[i] = std::max(dim_a, dim_b);
    }
    return result;
}
/* index */
std::vector<size_t> Tensor::broadcastIdx(const std::vector<size_t>& idx, const std::vector<size_t>& org_shape) const{
    std::vector<size_t> out(org_shape.size());
    int offset = static_cast<int>(idx.size()) - static_cast<int>(org_shape.size());
    for(size_t i = 0; i < org_shape.size(); i ++ ){
        if(org_shape[i] == 1) {
            out[i] = 0;
        } else {
            size_t idx_pos = i + offset;
            if(offset < 0 && i < static_cast<size_t>(-offset)) {
                out[i] = 0;
            } else {
                out[i] = idx[idx_pos];
            }
        }
    }
    return out;
}
/* traverse */
void Tensor::traverse(const std::vector<size_t>& shape, std::function<void(const std::vector<size_t>&)> f) const {
    if (shape.empty()) {
        f(std::vector<size_t>());
        return;
    }
    std::vector<size_t> idx(shape.size(), 0);
    while(true){
        f(idx);
        int i = (int)shape.size() - 1;
        while(i >= 0){
            idx[i] ++;
            if(shape[i] > idx[i])break;
            idx[i] = 0;
            i --;
        }
        if(i < 0) break;
    }
}

/* backward function */
const std::vector<std::shared_ptr<Tensor>> Tensor::topSort() {
    std::vector<std::shared_ptr<Tensor>> res;
    std::set<std::shared_ptr<Tensor>> st;
    auto self = shared_from_this();
    std::function<void(std::shared_ptr<Tensor>)> dfs = [&](std::shared_ptr<Tensor> node) {
        if (st.count(node)) return;
        st.insert(node);
        for (const auto& p : node->prev) {
            dfs(p);
        }
        res.push_back(node);
    };
    dfs(self);
    std::reverse(res.begin(), res.end());
    return res;
}
void Tensor::backwardAll() {
    std::fill(grad.begin(), grad.end(), 1);
    for (const auto& t : topSort()) {
        t->backward();
    }
}