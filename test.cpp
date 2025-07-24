#include "./include/tensor.hpp"
#include <bits/stdc++.h>

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

void test_addition() {
    std::cout << "\n=== 测试加法运算 ===" << std::endl;
    // z = x + y, 期望梯度: dz/dx = 1, dz/dy = 1
    auto x = std::make_shared<Tensor>(std::vector<double>{1, 2}, std::vector<size_t>{2});
    auto y = std::make_shared<Tensor>(std::vector<double>{3, 4}, std::vector<size_t>{2});
    auto z = *x + y;
    
    print_tensor("x", x);
    print_tensor("y", y);
    print_tensor("z = x + y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [1, 1]
    print_grad("y", y);  // 期望: [1, 1]
    std::cout << "期望 x_grad: [1, 1], y_grad: [1, 1]" << std::endl;
}

void test_subtraction() {
    std::cout << "\n=== 测试减法运算 ===" << std::endl;
    // z = x - y, 期望梯度: dz/dx = 1, dz/dy = -1
    auto x = std::make_shared<Tensor>(std::vector<double>{5, 6}, std::vector<size_t>{2});
    auto y = std::make_shared<Tensor>(std::vector<double>{2, 3}, std::vector<size_t>{2});
    auto z = *x - y;
    
    print_tensor("x", x);
    print_tensor("y", y);
    print_tensor("z = x - y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [1, 1]
    print_grad("y", y);  // 期望: [-1, -1]
    std::cout << "期望 x_grad: [1, 1], y_grad: [-1, -1]" << std::endl;
}

void test_multiplication() {
    std::cout << "\n=== 测试乘法运算 ===" << std::endl;
    // z = x * y, 期望梯度: dz/dx = y, dz/dy = x
    auto x = std::make_shared<Tensor>(std::vector<double>{2, 3}, std::vector<size_t>{2});
    auto y = std::make_shared<Tensor>(std::vector<double>{4, 5}, std::vector<size_t>{2});
    auto z = *x * y;
    
    print_tensor("x", x);
    print_tensor("y", y);
    print_tensor("z = x * y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [4, 5] (y的值)
    print_grad("y", y);  // 期望: [2, 3] (x的值)
    std::cout << "期望 x_grad: [4, 5], y_grad: [2, 3]" << std::endl;
}

void test_division() {
    std::cout << "\n=== 测试除法运算 ===" << std::endl;
    // z = x / y, 期望梯度: dz/dx = 1/y, dz/dy = -x/(y^2)
    auto x = std::make_shared<Tensor>(std::vector<double>{8, 12}, std::vector<size_t>{2});
    auto y = std::make_shared<Tensor>(std::vector<double>{2, 3}, std::vector<size_t>{2});
    auto z = *x / y;
    
    print_tensor("x", x);
    print_tensor("y", y);
    print_tensor("z = x / y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [0.5, 0.333] (1/y)
    print_grad("y", y);  // 期望: [-2, -1.333] (-x/(y^2))
    std::cout << "期望 x_grad: [0.5, 0.333], y_grad: [-2, -1.333]" << std::endl;
}

void test_composite_operations() {
    std::cout << "\n=== 测试复合运算 ===" << std::endl;
    // z = (x + y) * x, 期望梯度计算
    auto x = std::make_shared<Tensor>(std::vector<double>{2, 3}, std::vector<size_t>{2});
    auto y = std::make_shared<Tensor>(std::vector<double>{1, 2}, std::vector<size_t>{2});
    
    auto temp = *x + y;  // temp = [3, 5]
    auto z = *temp * x;  // z = [6, 15]
    
    print_tensor("x", x);
    print_tensor("y", y);
    print_tensor("temp = x + y", temp);
    print_tensor("z = temp * x", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [5, 8] (dz/dx = temp + x = (x+y) + x = 2x+y)
    print_grad("y", y);  // 期望: [2, 3] (dz/dy = x)
    std::cout << "期望 x_grad: [5, 8], y_grad: [2, 3]" << std::endl;
    
    std::cout << "手动计算:" << std::endl;
    std::cout << "z = (x + y) * x" << std::endl;
    std::cout << "dz/dx = (x + y) + x = 2x + y = 2*[2,3] + [1,2] = [5,8]" << std::endl;
    std::cout << "dz/dy = x = [2,3]" << std::endl;
}

void test_your_original_case() {
    std::cout << "\n=== 测试你的原始案例 ===" << std::endl;
    // z = (x + y) * y
    auto x = std::make_shared<Tensor>(std::vector<double>{2, 3, 4, 5}, std::vector<size_t>{2, 2});
    auto y = std::make_shared<Tensor>(std::vector<double>{1, 2, 3, 4}, std::vector<size_t>{2, 2});
    auto z = *(*x + y) * y;
    
    print_tensor("x", x);
    print_tensor("y", y);
    print_tensor("z = (x + y) * y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [1, 2, 3, 4] (dz/dx = y)
    print_grad("y", y);  // 期望: [3, 5, 7, 9] (dz/dy = (x + y) + y = x + 2y)
    std::cout << "期望 x_grad: [1, 2, 3, 4], y_grad: [3, 5, 7, 9]" << std::endl;
    
    std::cout << "手动计算:" << std::endl;
    std::cout << "z = (x + y) * y" << std::endl;
    std::cout << "dz/dx = y = [1, 2, 3, 4]" << std::endl;
    std::cout << "dz/dy = (x + y) + y = x + 2y = [2,3,4,5] + 2*[1,2,3,4] = [4,7,10,13]" << std::endl;
}
void test_broadcast_basic() {
    std::cout << "\n=== 测试基础广播 ===" << std::endl;
    // 标量与向量广播: z = x + y
    auto x = std::make_shared<Tensor>(std::vector<double>{2, 3, 4}, std::vector<size_t>{3});     // [3]
    auto y = std::make_shared<Tensor>(std::vector<double>{5}, std::vector<size_t>{1});           // [1] 
    auto z = *x + y;
    
    print_tensor("x (shape [3])", x);
    print_tensor("y (shape [1])", y);
    print_tensor("z = x + y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [1, 1, 1]
    print_grad("y", y);  // 期望: [3] (求和)
    std::cout << "期望 x_grad: [1, 1, 1], y_grad: [3]" << std::endl;
}

void test_broadcast_matrix_vector() {
    std::cout << "\n=== 测试矩阵-向量广播 ===" << std::endl;
    // 矩阵与行向量广播: z = x + y
    auto x = std::make_shared<Tensor>(std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<size_t>{2, 3}); // [2,3]
    auto y = std::make_shared<Tensor>(std::vector<double>{10, 20, 30}, std::vector<size_t>{1, 3});        // [1,3]
    auto z = *x + y;
    
    print_tensor("x (shape [2,3])", x);
    print_tensor("y (shape [1,3])", y);
    print_tensor("z = x + y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [1, 1, 1, 1, 1, 1]
    print_grad("y", y);  // 期望: [2, 2, 2] (每列求和)
    std::cout << "期望 x_grad: [1, 1, 1, 1, 1, 1], y_grad: [2, 2, 2]" << std::endl;
}

void test_broadcast_column_vector() {
    std::cout << "\n=== 测试矩阵-列向量广播 ===" << std::endl;
    // 矩阵与列向量广播: z = x * y
    auto x = std::make_shared<Tensor>(std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<size_t>{2, 3}); // [2,3]
    auto y = std::make_shared<Tensor>(std::vector<double>{2, 3}, std::vector<size_t>{2, 1});             // [2,1]
    auto z = *x * y;
    
    print_tensor("x (shape [2,3])", x);
    print_tensor("y (shape [2,1])", y);
    print_tensor("z = x * y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [2, 2, 2, 3, 3, 3] (y的值广播)
    print_grad("y", y);  // 期望: [6, 15] (每行求和: 1+2+3=6, 4+5+6=15)
    std::cout << "期望 x_grad: [2, 2, 2, 3, 3, 3], y_grad: [6, 15]" << std::endl;
}

void test_broadcast_scalar_operations() {
    std::cout << "\n=== 测试标量广播混合运算 ===" << std::endl;
    // z = (x + scalar) * y
    auto x = std::make_shared<Tensor>(std::vector<double>{1, 2, 3, 4}, std::vector<size_t>{2, 2});  // [2,2]
    auto scalar = std::make_shared<Tensor>(std::vector<double>{10}, std::vector<size_t>{1});        // [1]
    auto y = std::make_shared<Tensor>(std::vector<double>{2, 3}, std::vector<size_t>{2, 1});        // [2,1]
    
    auto temp = *x + scalar;  // temp = [[11, 12], [13, 14]]
    auto z = *temp * y;       // z = [[22, 24], [39, 42]]
    
    print_tensor("x (shape [2,2])", x);
    print_tensor("scalar (shape [1])", scalar);
    print_tensor("y (shape [2,1])", y);
    print_tensor("temp = x + scalar", temp);
    print_tensor("z = temp * y", z);
    
    z->backwardAll();
    
    print_grad("x", x);      // 期望: [2, 2, 3, 3] (y的值广播)
    print_grad("scalar", scalar); // 期望: [5] (所有梯度求和: 2+2+3+3=10)
    print_grad("y", y);      // 期望: [23, 27] (每行temp值求和)
    
    std::cout << "手动计算:" << std::endl;
    std::cout << "z = (x + scalar) * y" << std::endl;
    std::cout << "dz/dx = y 广播 = [[2, 2], [3, 3]]" << std::endl;
    std::cout << "dz/d(scalar) = y 的所有元素求和 = [2+2+3+3] = [10]" << std::endl;
    std::cout << "dz/dy = (x + scalar) 按行求和 = [[11+12], [13+14]] = [23, 27]" << std::endl;
}

void test_broadcast_complex() {
    std::cout << "\n=== 测试复杂广播运算 ===" << std::endl;
    // z = (x + y) / (x - bias), 多重广播
    auto x = std::make_shared<Tensor>(std::vector<double>{4, 6, 8, 10, 12, 14}, std::vector<size_t>{2, 3}); // [2,3]
    auto y = std::make_shared<Tensor>(std::vector<double>{1, 2, 3}, std::vector<size_t>{1, 3});             // [1,3] 行向量
    auto bias = std::make_shared<Tensor>(std::vector<double>{2}, std::vector<size_t>{1});                   // [1] 标量
    
    auto sum_xy = *x + y;        // 广播加法
    auto diff_x_bias = *x - bias; // 广播减法
    auto z = *sum_xy / diff_x_bias; // 元素除法
    
    print_tensor("x (shape [2,3])", x);
    print_tensor("y (shape [1,3])", y);
    print_tensor("bias (shape [1])", bias);
    print_tensor("sum_xy = x + y", sum_xy);
    print_tensor("diff_x_bias = x - bias", diff_x_bias);
    print_tensor("z = sum_xy / diff_x_bias", z);
    
    z->backwardAll();
    
    print_grad("x", x);
    print_grad("y", y);
    print_grad("bias", bias);
    
    std::cout << "复杂梯度计算，需要应用链式法则和广播求和规则" << std::endl;
}

void test_broadcast_3d() {
    std::cout << "\n=== 测试3D张量广播 ===" << std::endl;
    // 3D张量与不同维度的广播
    auto x = std::make_shared<Tensor>(std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8}, std::vector<size_t>{2, 2, 2}); // [2,2,2]
    auto y = std::make_shared<Tensor>(std::vector<double>{10, 20}, std::vector<size_t>{1, 1, 2});                  // [1,1,2]
    auto z = *x + y;
    
    print_tensor("x (shape [2,2,2])", x);
    print_tensor("y (shape [1,1,2])", y);
    print_tensor("z = x + y", z);
    
    z->backwardAll();
    
    print_grad("x", x);  // 期望: [1, 1, 1, 1, 1, 1, 1, 1]
    print_grad("y", y);  // 期望: [4, 4] (每个广播维度求和)
    std::cout << "期望 x_grad: 全为1, y_grad: [4, 4]" << std::endl;
}
int main() {
    test_addition();
    test_subtraction();
    test_multiplication();
    test_division();
    test_composite_operations();
    test_your_original_case();
    test_broadcast_basic();
    test_broadcast_matrix_vector();
    test_broadcast_column_vector();
    test_broadcast_scalar_operations();
    test_broadcast_complex();
    test_broadcast_3d();
    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}