#pragma once

#include <memory>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "../../../aten/include/tensor.hpp"

namespace torch::autograd {
class Engine {

private:
    std::shared_ptr<at::Tensor> root;
    std::vector<std::shared_ptr<at::Tensor>> sorted;
    std::unordered_set<std::shared_ptr<at::Tensor>> st;

public:
    explicit Engine(std::shared_ptr<at::Tensor> _root);
    virtual ~Engine() = default;

    void topsort();
    void dfs(std::shared_ptr<at::Tensor> node);
    void execute();
};

}; 
