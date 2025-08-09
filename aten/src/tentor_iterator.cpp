#include "../include/tensor_iterator.hpp"

namespace at {

std::vector<size_t> broadcastShape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2){
    std::vector<size_t> res;
    size_t max_ndim = std::max(shape1.size(), shape2.size());
    res.resize(max_ndim);
    
    for(i64 i = max_ndim - 1; i >= 0; i--){
        i64 dim1_idx = (i64)shape1.size() - (max_ndim - i);
        i64 dim2_idx = (i64)shape2.size() - (max_ndim - i);
        
        size_t dim_a = dim1_idx >= 0 ? shape1[dim1_idx] : 1;
        size_t dim_b = dim2_idx >= 0 ? shape2[dim2_idx] : 1;
        
        if(dim_a != dim_b && dim_a != 1 && dim_b != 1) 
            throw std::out_of_range("Dimensions dismatch.");
        res[i] = std::max(dim_a, dim_b);
    }
    return res;
}

std::vector<size_t> transferIdx(const std::vector<size_t> &idx, const std::vector<size_t> &org_shape){
    std::vector<size_t>res(org_shape.size());
    i64 offset = (i64)idx.size() - (i64)org_shape.size();
    
    for(i64 i = (i64)res.size() - 1; i >= 0; i -- ){
        i64 idx_pos = i + offset;
        if(org_shape[i] != 1){
            if(idx_pos >= 0 && idx[idx_pos] >= org_shape[i]) 
                throw std::out_of_range("Idx is out of range.");
            res[i] = (idx_pos >= 0) ? idx[idx_pos] : 0;
        }else{
            res[i] = 0;
        }
    }
    return res;
}

void traverse(const std::vector<size_t>& shape, std::function<void(const std::vector<size_t>& idx)> f){
    std::function<void(size_t, std::vector<size_t>&)> dfs = [&](size_t depth, std::vector<size_t>& idx){
        if(depth == shape.size()){
            f(idx);
            return;
        }
        for(size_t i=0;i<shape[depth];i++){
            idx[depth] = i;
            dfs(depth+1, idx);
        }
    };
    std::vector<size_t> idx(shape.size(), 0);
    dfs(0, idx);
}

} // namespace at