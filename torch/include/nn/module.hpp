#pragma once
#include <memory>
#include <map>
#include <functional>
#include "../../../aten/include/tensor.hpp"

namespace torch::nn {

class Module : public std::enable_shared_from_this<Module>{
public:
    Module() = default;
    virtual ~Module() = default;

    virtual std::shared_ptr<at::Tensor> forward(const std::shared_ptr<at::Tensor> input) = 0;
    void register_parameter(const std::string& name, std::shared_ptr<at::Tensor> p);
    void register_module(const std::string& name, std::shared_ptr<Module> m);
    std::vector<std::shared_ptr<at::Tensor>> parameters(bool include_self=true);
    std::vector<std::shared_ptr<Module>> modules(bool include_self=false);

protected:
    std::vector<std::pair<std::string, std::shared_ptr<at::Tensor>>> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules_;
};

} // namespace torch::nn