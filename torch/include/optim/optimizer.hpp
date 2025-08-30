#pragma once

namespace torch::optim {
    
class Optimizer {
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;
    
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};

} // namespace torch::optim