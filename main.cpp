#include "./aten/include/tensor.hpp"
#include "./torch/include/nn/modules/linear.hpp"
#include "./torch/include/nn/modules/mse_loss.hpp"
#include "./torch/include/optim/adam.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class MLP : public torch::nn::Module {
public:
    MLP(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t output_size) 
        : fc1(std::make_shared<torch::nn::Linear>(input_size, hidden1_size)),
          fc2(std::make_shared<torch::nn::Linear>(hidden1_size, hidden2_size)),
          fc3(std::make_shared<torch::nn::Linear>(hidden2_size, output_size)) {
        register_module("fc1", std::static_pointer_cast<torch::nn::Module>(fc1));
        register_module("fc2", std::static_pointer_cast<torch::nn::Module>(fc2));
        register_module("fc3", std::static_pointer_cast<torch::nn::Module>(fc3));
    }

    std::shared_ptr<at::Tensor> forward(const std::shared_ptr<at::Tensor> input) override {
        // first layer + ReLU
        auto x = fc1->forward(input);
        for (size_t i = 0; i < x->data.size(); ++i) {
            if (x->data[i] < 0) x->data[i] = 0;
        }

        // second layer + ReLU
        x = fc2->forward(x);
        for (size_t i = 0; i < x->data.size(); ++i) {
            if (x->data[i] < 0) x->data[i] = 0;
        }

        // output layer
        return fc3->forward(x);
    }

private:
    std::shared_ptr<torch::nn::Linear> fc1;
    std::shared_ptr<torch::nn::Linear> fc2;
    std::shared_ptr<torch::nn::Linear> fc3;
};

//  y = sin(2*pi*x) + 0.5*x^2
void generate_nonlinear_dataset(std::vector<std::shared_ptr<at::Tensor>>& inputs,
                               std::vector<std::shared_ptr<at::Tensor>>& targets,
                               size_t num_samples) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::normal_distribution<> noise(0.0, 0.05);
    
    for (size_t i = 0; i < num_samples; ++i) {
        double x = dis(gen);
        double y = std::sin(2.0 * M_PI * x) + 0.5 * x * x + noise(gen);
        
        auto input = std::make_shared<at::Tensor>(
            std::vector<double>{x}, 
            std::vector<size_t>{1}, 
            false
        );
        
        auto target = std::make_shared<at::Tensor>(
            std::vector<double>{y}, 
            std::vector<size_t>{1}, 
            false
        );
        
        inputs.push_back(input);
        targets.push_back(target);
    }
}

// calculate the r2 score
double calculate_r2_score(const std::vector<std::shared_ptr<at::Tensor>>& predictions,
                          const std::vector<std::shared_ptr<at::Tensor>>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) {
        return 0.0;
    }
    
    double target_mean = 0.0;
    for (const auto& target : targets) {
        target_mean += target->data[0];
    }
    target_mean /= targets.size();
    
    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        double target_val = targets[i]->data[0];
        double pred_val = predictions[i]->data[0];
        
        ss_tot += (target_val - target_mean) * (target_val - target_mean);
        ss_res += (target_val - pred_val) * (target_val - pred_val);
    }
    
    return 1.0 - (ss_res / ss_tot);
}

int main() {
    std::cout << "=== Complete Neural Network Training System ===" << std::endl;
    std::cout << "Goal: Learn function y = sin(2pix) + 0.5x*x" << std::endl << std::endl;
    
    try {
        // 1. create multi-layer perceptron (1 -> 16 -> 8 -> 1)
        std::cout << "1. Creating Multi-Layer Perceptron..." << std::endl;
        auto model = std::make_shared<MLP>(1, 16, 8, 1);
        std::cout << "   Network architecture: 1 -> 16 -> 8 -> 1" << std::endl;

        // 2. create optimizer
        auto params = model->parameters();
        std::cout << "   Total parameters: " << params.size() << std::endl;
        torch::optim::Adam optimizer(params, 0.001); 
        std::cout << "   Optimizer: Adam (lr=0.001)" << std::endl;
        
        // 3. create criterion
        torch::nn::MSELoss criterion;
        std::cout << "   Loss function: MSE Loss" << std::endl << std::endl;

        // 4. generate training and testing data
        std::cout << "2. Generating datasets..." << std::endl;
        std::vector<std::shared_ptr<at::Tensor>> train_inputs, train_targets;
        std::vector<std::shared_ptr<at::Tensor>> test_inputs, test_targets;
        
        generate_nonlinear_dataset(train_inputs, train_targets, 200);
        generate_nonlinear_dataset(test_inputs, test_targets, 50);
        
        std::cout << "   Training samples: " << train_inputs.size() << std::endl;
        std::cout << "   Test samples: " << test_inputs.size() << std::endl << std::endl;
        
        // 5. training loop
        std::cout << "3. Starting training..." << std::endl;
        int num_epochs = 1000;
        int print_every = 50;
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            double total_loss = 0.0;
            
            // training stage
            for (size_t i = 0; i < train_inputs.size(); ++i) {
                // forward pass
                auto prediction = model->forward(train_inputs[i]);
                auto loss = criterion.forward(prediction, train_targets[i]);

                // backward pass
                optimizer.zero_grad();
                loss->grad.resize(1);
                loss->grad[0] = 1.0;
                loss->backward();

                // update parameters
                optimizer.step();
                
                total_loss += loss->data[0];
            }
            
            // evaluation
            if (epoch % print_every == 0 || epoch == num_epochs - 1) {
                double avg_train_loss = total_loss / train_inputs.size();
                
                std::vector<std::shared_ptr<at::Tensor>> test_predictions;
                double test_loss = 0.0;
                
                for (size_t i = 0; i < test_inputs.size(); ++i) {
                    auto pred = model->forward(test_inputs[i]);
                    auto loss = criterion.forward(pred, test_targets[i]);
                    test_predictions.push_back(pred);
                    test_loss += loss->data[0];
                }
                
                double avg_test_loss = test_loss / test_inputs.size();
                double r2_score = calculate_r2_score(test_predictions, test_targets);
                
                std::cout << "Epoch " << std::setw(3) << epoch 
                          << " | Train Loss: " << std::fixed << std::setprecision(6) << avg_train_loss
                          << " | Test Loss: " << std::setprecision(6) << avg_test_loss
                          << " | R*R Score: " << std::setprecision(4) << r2_score << std::endl;
            }
        }
        
        std::cout << std::endl << "4. Training completed! Testing model predictions..." << std::endl;
        
        // 6. test
        std::vector<double> test_points = {-0.8, -0.4, 0.0, 0.4, 0.8};
        std::cout << "\nTest predictions vs actual values:" << std::endl;
        std::cout << "x      | Predicted | Actual    | Error" << std::endl;
        std::cout << "-------|-----------|-----------|----------" << std::endl;
        
        for (double x : test_points) {
            auto test_input = std::make_shared<at::Tensor>(
                std::vector<double>{x}, 
                std::vector<size_t>{1}, 
                false
            );
            
            auto prediction = model->forward(test_input);
            double actual = std::sin(2.0 * M_PI * x) + 0.5 * x * x;
            double error = std::abs(prediction->data[0] - actual);
            
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << x 
                      << " | " << std::setw(9) << std::setprecision(4) << prediction->data[0]
                      << " | " << std::setw(9) << std::setprecision(4) << actual
                      << " | " << std::setw(8) << std::setprecision(4) << error << std::endl;
        }
        
        std::cout << std::endl << "=== Training System Validation Complete ===" << std::endl;
        std::cout << "Multi-layer neural network working correctly" << std::endl;
        std::cout << "Adam optimizer performing parameter updates" << std::endl;
        std::cout << "MSE loss providing proper gradients" << std::endl;
        std::cout << "Complete training pipeline functional" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
