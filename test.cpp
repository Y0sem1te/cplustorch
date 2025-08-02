#include "./include/tensor.hpp"
#include "./include/nn/functional.hpp"
#include "./include/nn/linear.hpp"
#include "./include/criterion/mse_loss.hpp"
#include "./include/criterion/cross_entropy_loss.hpp"
#include "./include/optimizer/adam.hpp"
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
    // std::cout << "\n=== Test transpose and matmul2d (complex) ===" << std::endl;
    // // A: 2x3, B: 3x2
    // std::vector<double> a_data = {1, 2, 3, 4, 5, 6}; // shape: [2,3]
    // std::vector<double> b_data = {7, 8, 9, 10, 11, 12}; // shape: [3,2]
    // auto A = std::make_shared<Tensor>(a_data, std::vector<size_t>{2,3});
    // auto B = std::make_shared<Tensor>(b_data, std::vector<size_t>{3,2});
    // A->requires_grad = true;
    // B->requires_grad = true;
    // // Transpose A: shape [3,2]
    // auto A_trans = A->transpose(0,1);
    // print_tensor("A_trans (A transposed)", A_trans);
    // // Matmul: C = A_trans x B, shape [3,2] x [3,2] (should throw shape error, so use B_trans)
    // auto B_trans = B->transpose(0,1); // shape [2,3]
    // print_tensor("B_trans (B transposed)", B_trans);
    // // Now matmul: [3,2] x [2,3] => [3,3]
    // auto C = A_trans->matmul2d(B_trans);
    // print_tensor("C (A_trans x B_trans)", C);
    // C->grad = std::vector<double>(C->data.size(), 1.0);
    // C->backwardAll();
    // print_grad("A_grad", A);
    // print_grad("A_trans_grad", A_trans);
    // print_grad("B_grad", B);
    // print_grad("B_trans_grad", B_trans);
    // print_grad("C_grad", C);
    // // More complex matrix chain operation
    // std::cout << "\n=== Test complex matrix chain operation ===" << std::endl;
    // // C (3x3) and E (3x3)
    // std::vector<double> c_data = {1,2,3,4,5,6,7,8,9};
    // std::vector<double> e_data = {2,1,3,1,2,1,3,1,2};
    // auto C2 = std::make_shared<Tensor>(c_data, std::vector<size_t>{3,3});
    // auto E = std::make_shared<Tensor>(e_data, std::vector<size_t>{3,3});
    // C2->requires_grad = true;
    // E->requires_grad = true;
    // // chain rule：D = tanh(A_trans.matmul2d(B_trans) + C2) * E
    // auto M = A_trans->matmul2d(B_trans); // [3,3]
    // auto S = *M + C2; // [3,3]
    // auto T = Functional::tanh(S); // [3,3]
    // auto D = *T * E; // [3,3]
    // print_tensor("D (final result)", D);
    // D->grad = std::vector<double>(D->data.size(), 1.0);
    // D->backwardAll();
    // print_grad("A_grad", A);
    // print_grad("A_trans_grad", A_trans);
    // print_grad("B_grad", B);
    // print_grad("B_trans_grad", B_trans);
    // print_grad("C2_grad", C2);
    // print_grad("E_grad", E);
    // print_grad("M_grad", M);
    // print_grad("S_grad", S);
    // print_grad("T_grad", T);
    // print_grad("D_grad", D);
    // Test Linear layer
    
    // Test Linear layer (deterministic)
    // std::cout << "\n=== Test Linear layer ===" << std::endl;
    // // Linear(3,2,true), weights = [[1,2,3],[4,5,6]], bias = [1,2]
    // auto linear = std::make_shared<Linear>(3, 2, true);
    // linear->weights->data = {1,2,3,4,5,6}; // shape [2,3], row-major
    // linear->bias->data = {1,2}; // shape [1,2]
    // linear->weights->requires_grad = true;
    // linear->bias->requires_grad = true;
    // auto input = std::make_shared<Tensor>(std::vector<double>{1,2,3,4,5,6}, std::vector<size_t>{2,3});
    // input->requires_grad = true;
    // auto output = (*linear)(input);
    // print_tensor("output", output);
    // // loss = sum(output)
    // output->grad = std::vector<double>(output->data.size(), 1.0);
    // output->backwardAll();
    // print_grad("input_grad", input);
    // print_grad("weights_grad", linear->weights);
    // print_grad("bias_grad", linear->bias);
    // print_grad("output_grad", output);
    // // output = input × weights^T + bias
    // // the first line: [1,2,3] × [1,2,3]^T = 1*1+2*2+3*3=14, [1,2,3] × [4,5,6]^T = 1*4+2*5+3*6=32
    // // with bias: [15,34]
    // // the second line: [4,5,6] × [1,2,3]^T = 4*1+5*2+6*3=32, [4,5,6] × [4,5,6]^T = 4*4+5*5+6*6=77
    // // with bias: [33,79]
    // // output: [15,34,33,79]
    // // Test chained Linear layers with activation
    // std::cout << "\n=== Test chained Linear layers with activation ===" << std::endl;
    // // Linear1: 3->4, Linear2: 4->2
    // auto linear1 = std::make_shared<Linear>(3, 4, true);
    // auto linear2 = std::make_shared<Linear>(4, 2, true);
    // linear1->weights->data = {1,2,3,4,5,6,7,8,9,10,11,12}; // [4,3]
    // linear1->bias->data = {1,2,3,4}; // [1,4]
    // linear2->weights->data = {1,2,3,4,5,6,7,8}; // [2,4]
    // linear2->bias->data = {1,2}; // [1,2]
    // linear1->weights->requires_grad = true;
    // linear1->bias->requires_grad = true;
    // linear2->weights->requires_grad = true;
    // linear2->bias->requires_grad = true;
    // auto input2 = std::make_shared<Tensor>(std::vector<double>{1,2,3,4,5,6}, std::vector<size_t>{2,3});
    // input2->requires_grad = true;
    // auto out1 = (*linear1)(input2);
    // auto act1 = Functional::tanh(out1);
    // auto out2 = (*linear2)(act1);
    // print_tensor("out2", out2);
    // // loss = sum(out2)
    // out2->grad = std::vector<double>(out2->data.size(), 1.0);
    // out2->backwardAll();
    // print_grad("input_grad", input2);
    // print_grad("linear1_weights_grad", linear1->weights);
    // print_grad("linear1_bias_grad", linear1->bias);
    // print_grad("linear2_weights_grad", linear2->weights);
    // print_grad("linear2_bias_grad", linear2->bias);
    // print_grad("out1_grad", out1);
    // print_grad("act1_grad", act1);
    // print_grad("out2_grad", out2);
    // === MSELoss test case ===
    // std::cout << "\n=== Test MSELoss ===" << std::endl;
    // auto mse = std::make_shared<MSELoss>();

    // // 1：output = [1,2,3], label = [1,2,3]，loss=0
    // auto out1 = std::make_shared<Tensor>(std::vector<double>{1,2,3}, std::vector<size_t>{3});
    // auto label1 = std::make_shared<Tensor>(std::vector<double>{1,2,3}, std::vector<size_t>{3});
    // out1->requires_grad = true;
    // label1->requires_grad = true;
    // auto loss1 = (*mse)(out1, label1);
    // print_tensor("loss1", loss1); // 0
    // loss1->grad = {1.0};
    // loss1->backwardAll();
    // print_grad("out1_grad", out1);    // [0,0,0]
    // print_grad("label1_grad", label1);// [0,0,0]

    // // 2：output = [2,4], label = [1,3]，loss=((2-1)^2 + (4-3)^2)/2 = (1+1)/2 = 1
    // auto out2 = std::make_shared<Tensor>(std::vector<double>{2,4}, std::vector<size_t>{2});
    // auto label2 = std::make_shared<Tensor>(std::vector<double>{1,3}, std::vector<size_t>{2});
    // out2->requires_grad = true;
    // label2->requires_grad = true;
    // auto loss2 = (*mse)(out2, label2);
    // print_tensor("loss2", loss2); // 1
    // loss2->grad = {1.0};
    // loss2->backwardAll();
    // print_grad("out2_grad", out2);    // [(2-1)*2/2=1, (4-3)*2/2=1]
    // print_grad("label2_grad", label2);// [-1, -1]

    // // 3：output = [0,0], label = [1,1]，loss=((0-1)^2 + (0-1)^2)/2 = (1+1)/2 = 1
    // auto out3 = std::make_shared<Tensor>(std::vector<double>{0,0}, std::vector<size_t>{2});
    // auto label3 = std::make_shared<Tensor>(std::vector<double>{1,1}, std::vector<size_t>{2});
    // out3->requires_grad = true;
    // label3->requires_grad = true;
    // auto loss3 = (*mse)(out3, label3);
    // print_tensor("loss3", loss3); // 1
    // loss3->grad = {1.0};
    // loss3->backwardAll();
    // print_grad("out3_grad", out3);    // [-1, -1]
    // print_grad("label3_grad", label3);// [1, 1]

    // // 4：output = [1,2,3,4], label = [4,3,2,1]，shape=[2,2]
    // // loss = ((1-4)^2 + (2-3)^2 + (3-2)^2 + (4-1)^2)/4 = (9+1+1+9)/4 = 20/4 = 5
    // // output_grad: [2*(1-4)/4= -1.5, 2*(2-3)/4= -0.5, 2*(3-2)/4=0.5, 2*(4-1)/4=1.5]
    // // label_grad:  [2*(4-1)/4=1.5, 2*(3-2)/4=0.5, 2*(2-3)/4=-0.5, 2*(1-4)/4=-1.5]
    // auto out4 = std::make_shared<Tensor>(std::vector<double>{1,2,3,4}, std::vector<size_t>{2,2});
    // auto label4 = std::make_shared<Tensor>(std::vector<double>{4,3,2,1}, std::vector<size_t>{2,2});
    // out4->requires_grad = true;
    // label4->requires_grad = true;
    // auto loss4 = (*mse)(out4, label4);
    // print_tensor("loss4", loss4); // 5
    // loss4->grad = {1.0};
    // loss4->backwardAll();
    // print_grad("out4_grad", out4);    // [-1.5, -0.5, 0.5, 1.5]
    // print_grad("label4_grad", label4);// [1.5, 0.5, -0.5, -1.5]

    // === CrossEntropyLoss test case ===
    std::cout << "\n=== Test CrossEntropyLoss ===" << std::endl;
    auto cross_entropy = std::make_shared<CrossEntropyLoss>();

    // 1：output = [0.7, 0.2, 0.1], label = [1,0,0] (one-hot)
    // loss = -[1*log(0.7) + 0*log(0.2) + 0*log(0.1)] ≈ -log(0.7) ≈ 0.3567
    auto ce_out1 = std::make_shared<Tensor>(std::vector<double>{0.7, 0.2, 0.1}, std::vector<size_t>{3});
    auto ce_label1 = std::make_shared<Tensor>(std::vector<double>{1,0,0}, std::vector<size_t>{3});
    ce_out1->requires_grad = true;
    ce_label1->requires_grad = true;
    auto ce_loss1 = (*cross_entropy)(ce_out1, ce_label1);
    print_tensor("ce_loss1", ce_loss1); // ≈0.3567
    ce_loss1->grad = {1.0};
    ce_loss1->backwardAll();
    print_grad("ce_out1_grad", ce_out1);    // [ -1/0.7, 0, 0 ] ≈ [-1.4286, 0, 0]
    print_grad("ce_label1_grad", ce_label1);// [ -log(0.7), -log(0.2), -log(0.1) ] ≈ [ -0.3567, -1.6094, -2.3026 ]

    // 例2：output = [0.1, 0.8, 0.1], label = [0,1,0]
    // loss = -log(0.8) ≈ 0.2231
    auto ce_out2 = std::make_shared<Tensor>(std::vector<double>{0.1, 0.8, 0.1}, std::vector<size_t>{3});
    auto ce_label2 = std::make_shared<Tensor>(std::vector<double>{0,1,0}, std::vector<size_t>{3});
    ce_out2->requires_grad = true;
    ce_label2->requires_grad = true;
    auto ce_loss2 = (*cross_entropy)(ce_out2, ce_label2);
    print_tensor("ce_loss2", ce_loss2); // ≈0.2231
    ce_loss2->grad = {1.0};
    ce_loss2->backwardAll();
    print_grad("ce_out2_grad", ce_out2);    // [0, -1/0.8, 0] ≈ [0, -1.25, 0]
    print_grad("ce_label2_grad", ce_label2);// [ -log(0.1), -log(0.8), -log(0.1) ] ≈ [2.3026, -0.2231, 2.3026]

    // === Adam Optimizer correctness test ===
    std::cout << "\n=== Test Adam Optimizer ===" << std::endl;
    auto param = std::make_shared<Tensor>(std::vector<double>{0.0}, std::vector<size_t>{1});
    param->requires_grad = true;
    Adam adam(param, 0.1);

    for (int step = 0; step < 100; ++step) {
        auto loss = std::make_shared<Tensor>(std::vector<double>{(param->data[0] - 3.0) * (param->data[0] - 3.0)}, std::vector<size_t>{1});
        loss->requires_grad = true;
        param->grad[0] = 2.0 * (param->data[0] - 3.0);
        adam.step();
        adam.zero_grad();
        std::cout << "Step " << step << ": x = " << param->data[0] << ", loss = " << loss->data[0] << std::endl;
    }

    return 0;
}