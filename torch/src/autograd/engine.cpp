#include "../../include/autograd/engine.hpp"
#include "../../../aten/include/tensor.hpp"

namespace at::autograd {

Engine::Engine(std::shared_ptr<Tensor> _root) :root(_root){}

void Engine::dfs(std::shared_ptr<Tensor> node){
    if(st.count(node)) return;
    st.insert(node);
    for(auto it: node->prev){
        dfs(it);
    }
    sorted.push_back(node);
}

void Engine::topsort(){
    st.clear();
    sorted.clear();
    dfs(root);
    std::reverse(sorted.begin(), sorted.end());
}

void Engine::execute(){
    // Initialize root gradient to 1
    std::fill(root->grad.begin(), root->grad.end(), 1.0);
    
    for(auto it: sorted){
        it->backward_it();
    }
}

}; // namespace autograd