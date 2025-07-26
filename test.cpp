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
    // std::cout << "\n=== Complex expression with require_grad control ===" << std::endl;
    // auto a = std::make_shared<Tensor>(std::vector<double>{2.0}, std::vector<size_t>{1});
    // auto b = std::make_shared<Tensor>(std::vector<double>{3.0}, std::vector<size_t>{1});
    // auto c = std::make_shared<Tensor>(std::vector<double>{8.0}, std::vector<size_t>{1});
    // auto d = std::make_shared<Tensor>(std::vector<double>{4.0}, std::vector<size_t>{1});
    // auto e = std::make_shared<Tensor>(std::vector<double>{1.0}, std::vector<size_t>{1});
    // a->requires_grad = true;
    // b->requires_grad = false;
    // c->requires_grad = true;
    // d->requires_grad = false;
    // e->requires_grad = false;
    // auto t1 = *a * b;
    // auto t2 = *c / d;
    // auto t3 = *t1 + t2;
    // auto t4 = *t3 - e;
    // auto z = Functional::tanh(t4);
    // print_tensor("z (result)", z);
    // z->backwardAll();
    // std::cout << "a requires_grad: " << a->requires_grad << std::endl;
    // print_grad("a_grad", a);
    // std::cout << "b requires_grad: " << b->requires_grad << std::endl;
    // print_grad("b_grad", b);
    // std::cout << "c requires_grad: " << c->requires_grad << std::endl;
    // print_grad("c_grad", c);
    // std::cout << "d requires_grad: " << d->requires_grad << std::endl;
    // print_grad("d_grad", d);
    // std::cout << "e requires_grad: " << e->requires_grad << std::endl;
    // print_grad("e_grad", e);
    // print_grad("z_grad", z);
    // Test transpose and matmul2d (complex)
    std::cout << "\n=== Test transpose and matmul2d (complex) ===" << std::endl;
    // A: 2x3, B: 3x2
    std::vector<double> a_data = {1, 2, 3, 4, 5, 6}; // shape: [2,3]
    std::vector<double> b_data = {7, 8, 9, 10, 11, 12}; // shape: [3,2]
    auto A = std::make_shared<Tensor>(a_data, std::vector<size_t>{2,3});
    auto B = std::make_shared<Tensor>(b_data, std::vector<size_t>{3,2});
    A->requires_grad = true;
    B->requires_grad = true;
    // Transpose A: shape [3,2]
    auto A_trans = A->transpose(0,1);
    print_tensor("A_trans (A transposed)", A_trans);
    // Matmul: C = A_trans x B, shape [3,2] x [3,2] (should throw shape error, so use B_trans)
    auto B_trans = B->transpose(0,1); // shape [2,3]
    print_tensor("B_trans (B transposed)", B_trans);
    // Now matmul: [3,2] x [2,3] => [3,3]
    auto C = A_trans->matmul2d(B_trans);
    print_tensor("C (A_trans x B_trans)", C);
    // 假设C的loss为所有元素之和
    C->grad = std::vector<double>(C->data.size(), 1.0);
    C->backwardAll();
    print_grad("A_grad", A);
    print_grad("A_trans_grad", A_trans);
    print_grad("B_grad", B);
    print_grad("B_trans_grad", B_trans);
    print_grad("C_grad", C);
    // More complex matrix chain operation
    std::cout << "\n=== Test complex matrix chain operation ===" << std::endl;
    // 构造 C (3x3) 和 E (3x3)
    std::vector<double> c_data = {1,2,3,4,5,6,7,8,9};
    std::vector<double> e_data = {2,1,3,1,2,1,3,1,2};
    auto C2 = std::make_shared<Tensor>(c_data, std::vector<size_t>{3,3});
    auto E = std::make_shared<Tensor>(e_data, std::vector<size_t>{3,3});
    C2->requires_grad = true;
    E->requires_grad = true;
    // 链式运算：D = tanh(A_trans.matmul2d(B_trans) + C2) * E
    auto M = A_trans->matmul2d(B_trans); // [3,3]
    auto S = *M + C2; // [3,3]
    auto T = Functional::tanh(S); // [3,3]
    auto D = *T * E; // [3,3]
    print_tensor("D (final result)", D);
    // 假设D的loss为所有元素之和
    D->grad = std::vector<double>(D->data.size(), 1.0);
    D->backwardAll();
    print_grad("A_grad", A);
    print_grad("A_trans_grad", A_trans);
    print_grad("B_grad", B);
    print_grad("B_trans_grad", B_trans);
    print_grad("C2_grad", C2);
    print_grad("E_grad", E);
    print_grad("M_grad", M);
    print_grad("S_grad", S);
    print_grad("T_grad", T);
    print_grad("D_grad", D);
    return 0;
}