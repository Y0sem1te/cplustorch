#include "../../include/native/ops.hpp"
#include "../../include/tensor.hpp"
#include "../../include/tensor_iterator.hpp"

namespace at::native {
    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other){
        std::vector<size_t> shape = at::broadcastShape(self->shape, other->shape);
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        std::vector<double> data(size, 0.0);
        at::traverse(shape, [&](const std::vector<size_t>& idx){
            size_t flattened_idx = 0;
            size_t stride = 1;
            for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                flattened_idx += idx[i] * stride;
                stride *= shape[i];
            }
            data[flattened_idx] = (*self)(at::transferIdx(idx, self->shape)) + (*other)(at::transferIdx(idx, other->shape));
        });
        auto res = std::make_shared<Tensor>(data, shape, self->require_grad || other->require_grad);
        if(self->require_grad) res->prev.insert(self);
        if(other->require_grad) res->prev.insert(other);
        res->backward_it = [self, other, res](){
            at::traverse(res->shape, [&](const std::vector<size_t>& idx){  
                if(!self->require_grad && !other->require_grad) return;          
                size_t flattened_idx = 0;
                size_t stride = 1;
                for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                    flattened_idx += idx[i] * stride;
                    stride *= res->shape[i];
                }
                if(self->require_grad) {
                    std::vector<size_t> self_idx = at::transferIdx(idx, self->shape);
                    size_t self_flattened_idx = 0;
                    size_t self_stride = 1;
                    for(i64 i = (i64)self_idx.size() - 1; i >= 0; i--){
                        self_flattened_idx += self_idx[i] * self_stride;
                        self_stride *= self->shape[i];
                    }
                    self->grad[self_flattened_idx] += res->grad[flattened_idx];
                }
                if(other->require_grad) {
                    std::vector<size_t> other_idx = at::transferIdx(idx, other->shape);
                    size_t other_flattened_idx = 0;
                    size_t other_stride = 1;
                    for(i64 i = (i64)other_idx.size() - 1; i >= 0; i--){
                        other_flattened_idx += other_idx[i] * other_stride;
                        other_stride *= other->shape[i];
                    }
                    other->grad[other_flattened_idx] += res->grad[flattened_idx];
                }
            });
        };
        return res;
    }
    std::shared_ptr<Tensor> negative(std::shared_ptr<Tensor> self){
        std::vector<size_t> shape = self->shape;
        size_t size = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<double> data(size, 0.0);
        at::traverse(shape, [&](const std::vector<size_t>& idx){
            size_t flattened_idx = 0;
            size_t stride = 1;
            for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                flattened_idx += idx[i] * stride;
                stride *= shape[i];
            }
            data[flattened_idx] = -(*self)(at::transferIdx(idx, self->shape));
        });
        auto res = std::make_shared<Tensor>(data, shape, self->require_grad);
        if(self->require_grad) res->prev.insert(self);
        res->backward_it = [self, res](){
            if(self->require_grad == false) return;
            at::traverse(res->shape, [&](const std::vector<size_t>& idx){            
                size_t flattened_idx = 0;
                size_t stride = 1;
                for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                    flattened_idx += idx[i] * stride;
                    stride *= res->shape[i];
                }
                if(self->require_grad) {
                    std::vector<size_t> self_idx = at::transferIdx(idx, self->shape);
                    size_t self_flattened_idx = 0;
                    size_t self_stride = 1;
                    for(i64 i = (i64)self_idx.size() - 1; i >= 0; i--){
                        self_flattened_idx += self_idx[i] * self_stride;
                        self_stride *= self->shape[i];
                    }
                    self->grad[self_flattened_idx] -= res->grad[flattened_idx];
                }
            });
        };
        return res;
    }
    std::shared_ptr<Tensor> minus(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other){
        return add(self, negative(other));
    }
    std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other){
        std::vector<size_t> shape = at::broadcastShape(self->shape, other->shape);
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        std::vector<double> data(size, 0.0);
        at::traverse(shape, [&](const std::vector<size_t>& idx){
            size_t flattened_idx = 0;
            size_t stride = 1;
            for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                flattened_idx += idx[i] * stride;
                stride *= shape[i];
            }
            data[flattened_idx] = (*self)(at::transferIdx(idx, self->shape)) * (*other)(at::transferIdx(idx, other->shape));
        });
        auto res = std::make_shared<Tensor>(data, shape, self->require_grad || other->require_grad);
        if(self->require_grad) res->prev.insert(self);
        if(other->require_grad) res->prev.insert(other);
        res->backward_it = [self, other, res](){
            at::traverse(res->shape, [&](const std::vector<size_t>& idx){  
                if(!self->require_grad && !other->require_grad) return;          
                size_t flattened_idx = 0;
                size_t stride = 1;
                for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                    flattened_idx += idx[i] * stride;
                    stride *= res->shape[i];
                }
                if(self->require_grad) {
                    std::vector<size_t> self_idx = at::transferIdx(idx, self->shape);
                    size_t self_flattened_idx = 0;
                    size_t self_stride = 1;
                    for(i64 i = (i64)self_idx.size() - 1; i >= 0; i--){
                        self_flattened_idx += self_idx[i] * self_stride;
                        self_stride *= self->shape[i];
                    }
                    self->grad[self_flattened_idx] += res->grad[flattened_idx] * (*other)(at::transferIdx(idx, other->shape));
                }
                if(other->require_grad) {
                    std::vector<size_t> other_idx = at::transferIdx(idx, other->shape);
                    size_t other_flattened_idx = 0;
                    size_t other_stride = 1;
                    for(i64 i = (i64)other_idx.size() - 1; i >= 0; i--){
                        other_flattened_idx += other_idx[i] * other_stride;
                        other_stride *= other->shape[i];
                    }
                    other->grad[other_flattened_idx] += res->grad[flattened_idx] * (*self)(at::transferIdx(idx, self->shape));
                }
            });
        };
        return res;
    }
    std::shared_ptr<Tensor> divide(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other){
        return multiply(self, other->pow(-1.0));
    }
    std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> self, double v){
        std::vector<size_t> shape = self->shape;
        size_t size = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<double> data(size, 0.0);
        at::traverse(shape, [&](const std::vector<size_t>& idx){
            size_t flattened_idx = 0;
            size_t stride = 1;
            for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                flattened_idx += idx[i] * stride;
                stride *= shape[i];
            }
            data[flattened_idx] = std::pow((*self)(at::transferIdx(idx, self->shape)), v);
        });
        auto res = std::make_shared<Tensor>(data, shape, self->require_grad);
        if(self->require_grad) res->prev.insert(self);
        res->backward_it = [self, res, v](){
            if(self->require_grad == false) return;
            at::traverse(res->shape, [&](const std::vector<size_t>& idx){            
                size_t flattened_idx = 0;
                size_t stride = 1;
                for(i64 i = (i64)idx.size() - 1; i >= 0; i--){
                    flattened_idx += idx[i] * stride;
                    stride *= res->shape[i];
                }
                if(self->require_grad) {
                    std::vector<size_t> self_idx = at::transferIdx(idx, self->shape);
                    size_t self_flattened_idx = 0;
                    size_t self_stride = 1;
                    for(i64 i = (i64)self_idx.size() - 1; i >= 0; i--){
                        self_flattened_idx += self_idx[i] * self_stride;
                        self_stride *= self->shape[i];
                    }
                    self->grad[self_flattened_idx] += v * res->grad[flattened_idx] * std::pow((*self)(at::transferIdx(idx, self->shape)), v - 1);
                }
            });
        };
        return res;
    }

    std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> self, size_t dim0, size_t dim1){
        if (self->shape.size() < 2)
            throw std::invalid_argument("Transpose is defined for at least 2D tensors");
        if (dim0 >= self->shape.size() || dim1 >= self->shape.size())
            throw std::out_of_range("Transpose indices out of range");
    
        std::vector<size_t> new_shape = self->shape;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::vector<size_t> new_stride(new_shape.size());
        new_stride[new_shape.size() - 1] = 1;
        for (int i = (int)new_shape.size() - 2; i >= 0; i--)
        {
            new_stride[i] = new_stride[i + 1] * new_shape[i + 1];
        }
        std::vector<double> res(self->data.size());
        at::traverse(self->shape, [&](std::vector<size_t> org_idx)
                            {
            size_t flatten_idx = 0;
            size_t stride = 1;
            for (int i = (int)self->shape.size() - 1; i >= 0; i--){
                flatten_idx += org_idx[i] * stride;
                stride *= self->shape[i];
            }
            std::swap(org_idx[dim0], org_idx[dim1]);
            size_t new_flatten_idx = 0;
            for(size_t i=0;i<new_stride.size();i++){
                new_flatten_idx += org_idx[i] * new_stride[i];
            }
            res[new_flatten_idx] = self->data[flatten_idx]; });
        auto out = std::make_shared<Tensor>(res, new_shape, self->require_grad);
        if(self->require_grad) out->prev.insert(self);
        out->backward_it = [self, dim0, dim1, out, new_shape, new_stride]()
        {
            if (self->require_grad == false)
                return;
            at::traverse(out->shape, [self, dim0, dim1, out](std::vector<size_t> out_idx)
                        {
            std::vector<size_t> self_idx = out_idx;
            std::swap(self_idx[dim0], self_idx[dim1]);
            size_t self_flatten_idx = 0;
            size_t self_stride = 1;
            for (int i = (int)self->shape.size() - 1; i >= 0; i--){
                self_flatten_idx += self_idx[i] * self_stride;
                self_stride *= self->shape[i];
            }
            size_t out_flatten_idx = 0;
            size_t out_stride = 1;
            for (int i = (int)out->shape.size() - 1; i >= 0; i--){
                out_flatten_idx += out_idx[i] * out_stride;
                out_stride *= out->shape[i];
            }
            self->grad[self_flatten_idx] += out->grad[out_flatten_idx]; });
        };
        return out;
    }
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other){
        // Handle 1D tensors
        bool self_is_1d = (self->shape.size() == 1);
        bool other_is_1d = (other->shape.size() == 1);
        
        // Create working shapes (promote 1D to 2D for computation)
        std::vector<size_t> self_work_shape = self->shape;
        std::vector<size_t> other_work_shape = other->shape;
        
        if (self_is_1d) {
            self_work_shape = {1, self->shape[0]}; // [n] -> [1, n]
        }
        if (other_is_1d) {
            other_work_shape = {other->shape[0], 1}; // [n] -> [n, 1]
        }
        
        if (self_work_shape.back() != other_work_shape[other_work_shape.size() - 2])
            throw std::invalid_argument("Shapes are not aligned for matrix multiplication");
        
        size_t self_rows = self_work_shape[self_work_shape.size() - 2];
        size_t self_cols = self_work_shape[self_work_shape.size() - 1];
        size_t other_rows = other_work_shape[other_work_shape.size() - 2];
        size_t other_cols = other_work_shape[other_work_shape.size() - 1];
        
        std::vector<size_t> self_batch_dims(self_work_shape.begin(), self_work_shape.end() - 2);
        std::vector<size_t> other_batch_dims(other_work_shape.begin(), other_work_shape.end() - 2);
        std::vector<size_t> batch_shape = at::broadcastShape(self_batch_dims, other_batch_dims);
        
        std::vector<size_t> out_shape = batch_shape;
        if (!self_is_1d) out_shape.push_back(self_rows);
        if (!other_is_1d) out_shape.push_back(other_cols);
        
        // Handle scalar result (both 1D with same length)
        if (self_is_1d && other_is_1d) {
            out_shape.push_back(1); // scalar result as [1]
        }
        
        size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<double> res(out_size, 0.0);
    
        size_t batch_size = std::accumulate(batch_shape.begin(), batch_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        
        for (size_t batch = 0; batch < batch_size; batch++) {
            std::vector<size_t> batch_idx(batch_shape.size());
            size_t temp_batch = batch;
            for (int i = (int)batch_shape.size() - 1; i >= 0; i--) {
                batch_idx[i] = temp_batch % batch_shape[i];
                temp_batch /= batch_shape[i];
            }

            std::vector<size_t> self_batch_idx = at::transferIdx(batch_idx, self_batch_dims);
            std::vector<size_t> other_batch_idx = at::transferIdx(batch_idx, other_batch_dims);

            size_t self_batch_offset = 0;
            size_t self_batch_stride = 1;
            for (int i = (int)self_batch_idx.size() - 1; i >= 0; i--) {
                self_batch_offset += self_batch_idx[i] * self_batch_stride;
                self_batch_stride *= self_batch_dims[i];
            }
            self_batch_offset *= self_rows * self_cols;

            size_t other_batch_offset = 0;
            size_t other_batch_stride = 1;
            for (int i = (int)other_batch_idx.size() - 1; i >= 0; i--) {
                other_batch_offset += other_batch_idx[i] * other_batch_stride;
                other_batch_stride *= other_batch_dims[i];
            }
            other_batch_offset *= other_rows * other_cols;

            size_t out_batch_offset = batch * self_rows * other_cols;

            for (size_t i = 0; i < self_rows; i++) {
                for (size_t j = 0; j < other_cols; j++) {
                    for (size_t k = 0; k < self_cols; k++) {
                        res[out_batch_offset + i * other_cols + j] += 
                            self->data[self_batch_offset + i * self_cols + k] * 
                            other->data[other_batch_offset + k * other_cols + j];
                    }
                }
            }
        }

        auto out = std::make_shared<Tensor>(res, out_shape, self->require_grad || other->require_grad);
        if(self->require_grad) out->prev.insert(self);
        if(other->require_grad) out->prev.insert(other);

        out->backward_it = [batch_shape, self_batch_dims, other_batch_dims, self_rows, self_cols, other_rows, other_cols, self, other, out](){
            if (self->require_grad == false && other->require_grad == false)
                return;
            
            size_t batch_size = std::accumulate(batch_shape.begin(), batch_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
            
            for (size_t batch = 0; batch < batch_size; batch++) {
                std::vector<size_t> batch_idx(batch_shape.size());
                size_t temp_batch = batch;
                for (int i = (int)batch_shape.size() - 1; i >= 0; i--) {
                    batch_idx[i] = temp_batch % batch_shape[i];
                    temp_batch /= batch_shape[i];
                }

                std::vector<size_t> self_batch_idx = at::transferIdx(batch_idx, self_batch_dims);
                std::vector<size_t> other_batch_idx = at::transferIdx(batch_idx, other_batch_dims);

                size_t self_batch_offset = 0;
                size_t self_batch_stride = 1;
                for (int i = (int)self_batch_idx.size() - 1; i >= 0; i--) {
                    self_batch_offset += self_batch_idx[i] * self_batch_stride;
                    self_batch_stride *= self_batch_dims[i];
                }
                self_batch_offset *= self_rows * self_cols;

                size_t other_batch_offset = 0;
                size_t other_batch_stride = 1;
                for (int i = (int)other_batch_idx.size() - 1; i >= 0; i--) {
                    other_batch_offset += other_batch_idx[i] * other_batch_stride;
                    other_batch_stride *= other_batch_dims[i];
                }
                other_batch_offset *= other_rows * other_cols;

                size_t out_batch_offset = batch * self_rows * other_cols;

                // dL/dA = dL/dC * B^T
                if (self->require_grad) {
                    for (size_t i = 0; i < self_rows; i++) {
                        for (size_t j = 0; j < self_cols; j++) {
                            double grad_sum = 0.0;
                            for (size_t k = 0; k < other_cols; k++) {
                                grad_sum += out->grad[out_batch_offset + i * other_cols + k] * 
                                           other->data[other_batch_offset + j * other_cols + k];
                            }
                            self->grad[self_batch_offset + i * self_cols + j] += grad_sum;
                        }
                    }
                }

                // dL/dB = A^T * dL/dC
                if (other->require_grad) {
                    for (size_t i = 0; i < other_rows; i++) {
                        for (size_t j = 0; j < other_cols; j++) {
                            double grad_sum = 0.0;
                            for (size_t k = 0; k < self_rows; k++) {
                                grad_sum += self->data[self_batch_offset + k * self_cols + i] * 
                                           out->grad[out_batch_offset + k * other_cols + j];
                            }
                            other->grad[other_batch_offset + i * other_cols + j] += grad_sum;
                        }
                    }
                }
            }
        };
        return out;
    }

} // namespace native
