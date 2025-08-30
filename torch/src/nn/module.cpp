#include "../../include/nn/module.hpp"
#include "../../../aten/include/tensor.hpp"

namespace torch::nn {

void Module::register_parameter(const std::string& name, std::shared_ptr<at::Tensor> p){
    parameters_.emplace_back(name, p);
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> m){
    modules_.emplace_back(name, m);
}

std::vector<std::shared_ptr<at::Tensor>> Module::parameters(bool include_self) {
    std::vector<std::shared_ptr<at::Tensor>> res;
    if(include_self){
        for(const auto& kv : parameters_){
            res.push_back(kv.second);
        }
    }
    for(const auto &kv: modules_){
        auto sub_res= kv.second->parameters(true);
        res.insert(res.end(), sub_res.begin(), sub_res.end());
    }
    return res;
}

std::vector<std::shared_ptr<Module>> Module::modules(bool include_self){
    std::vector<std::shared_ptr<Module>> res;
    if(include_self) {
        res.push_back(shared_from_this());
    }
    for (const auto& kv : modules_) {
        auto sub_res = kv.second->modules(true);
        res.insert(res.end(), sub_res.begin(), sub_res.end());
    }
    return res;
}

} // namespace torch::nn