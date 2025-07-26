#include "./include/tensor.hpp"
#include <bits/stdc++.h>
#include "./include/functional.hpp"

void print_tensor(const std::string& name, std::shared_ptr<Tensor> t) {
    std::cout << name << " data: ";
    for (const auto& val : t->data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void print_grad(const std::string& name, std::shared_ptr<Tensor> t) {
    std::cout << name << " grad: ";
    for (const auto& val : t->grad) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    // auto x1 = std::make_shared<Tensor>(std::vector<double>{2.0}, std::vector<size_t>{1});
    // auto x2 = std::make_shared<Tensor>(std::vector<double>{0.0}, std::vector<size_t>{1});
    // auto w1 = std::make_shared<Tensor>(std::vector<double>{-3.0}, std::vector<size_t>{1});
    // auto w2 = std::make_shared<Tensor>(std::vector<double>{1.0}, std::vector<size_t>{1});
    // auto b = std::make_shared<Tensor>(std::vector<double>{6.8813735870195432}, std::vector<size_t>{1});
    // auto o = *(*(*x1 * w1) + (*x2 * w2)) + b;
    // auto y = Functional::tanh(o);
    // print_tensor("y", y);
    // y->backwardAll();
    // print_grad("y_grad", y);
    // print_grad("x1_grad", x1);
    // print_grad("x2_grad", x2);
    // print_grad("w1_grad", w1);
    // print_grad("w2_grad", w2);
    // Test require_grad=false
    // std::cout << "\n=== Test require_grad ===" << std::endl;
    // auto a = std::make_shared<Tensor>(std::vector<double>{3.0}, std::vector<size_t>{1});
    // auto b = std::make_shared<Tensor>(std::vector<double>{4.0}, std::vector<size_t>{1});
    // a->requires_grad = false;
    // b->requires_grad = true;
    // auto z = *a + b;
    // z->backwardAll();
    // print_grad("a_grad (should be 0)", a);
    // print_grad("b_grad (should be 1)", b);
    // // Test require_grad=true
    // auto c = std::make_shared<Tensor>(std::vector<double>{5.0}, std::vector<size_t>{1});
    // c->requires_grad = true;
    // auto d = std::make_shared<Tensor>(std::vector<double>{6.0}, std::vector<size_t>{1});
    // d->requires_grad = true;
    // auto z2 = *c + d;
    // z2->backwardAll();
    // print_grad("c_grad (should be 1)", c);
    // print_grad("d_grad (should be 1)", d);
    // Complex require_grad test
    // std::cout << "\n=== Complex require_grad chain test ===" << std::endl;
    // auto x = std::make_shared<Tensor>(std::vector<double>{2.0}, std::vector<size_t>{1});
    // auto y = std::make_shared<Tensor>(std::vector<double>{3.0}, std::vector<size_t>{1});
    // auto w = std::make_shared<Tensor>(std::vector<double>{4.0}, std::vector<size_t>{1});
    // auto b = std::make_shared<Tensor>(std::vector<double>{5.0}, std::vector<size_t>{1});
    // auto v = std::make_shared<Tensor>(std::vector<double>{6.0}, std::vector<size_t>{1});
    // auto z = *(*x * y) + y;
    // x->requires_grad = false;
    // z->backwardAll();
    // print_grad("x_grad", x);
    // print_grad("y_grad", y);
    // print_grad("z_grad", z);
    // Complex expression: z = tanh((a * b + c / d) - e)
    std::cout << "\n=== Complex expression with require_grad control ===" << std::endl;
    auto a = std::make_shared<Tensor>(std::vector<double>{2.0}, std::vector<size_t>{1});
    auto b = std::make_shared<Tensor>(std::vector<double>{3.0}, std::vector<size_t>{1});
    auto c = std::make_shared<Tensor>(std::vector<double>{8.0}, std::vector<size_t>{1});
    auto d = std::make_shared<Tensor>(std::vector<double>{4.0}, std::vector<size_t>{1});
    auto e = std::make_shared<Tensor>(std::vector<double>{1.0}, std::vector<size_t>{1});
    a->requires_grad = true;
    b->requires_grad = false;
    c->requires_grad = true;
    d->requires_grad = false;
    e->requires_grad = false;
    auto t1 = *a * b;
    auto t2 = *c / d;
    auto t3 = *t1 + t2;
    auto t4 = *t3 - e;
    auto z = Functional::tanh(t4);
    print_tensor("z (result)", z);
    z->backwardAll();
    std::cout << "a requires_grad: " << a->requires_grad << std::endl;
    print_grad("a_grad", a);
    std::cout << "b requires_grad: " << b->requires_grad << std::endl;
    print_grad("b_grad", b);
    std::cout << "c requires_grad: " << c->requires_grad << std::endl;
    print_grad("c_grad", c);
    std::cout << "d requires_grad: " << d->requires_grad << std::endl;
    print_grad("d_grad", d);
    std::cout << "e requires_grad: " << e->requires_grad << std::endl;
    print_grad("e_grad", e);
    print_grad("z_grad", z);
    return 0;
}