#pragma once

#include <memory>
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace at {
class Tensor;

namespace autograd {

class Engine {

private:
    std::shared_ptr<Tensor> root;
    std::vector<std::shared_ptr<Tensor>> sorted;
    std::unordered_set<std::shared_ptr<Tensor>> st;

public:
    explicit Engine(std::shared_ptr<Tensor> _root);
    virtual ~Engine() = default;

    void topsort();
    void dfs(std::shared_ptr<Tensor> node);
    void execute();

};

}; // at
}; // autograd