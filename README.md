# CPlusTorch

🔥 A lightweight PyTorch implementation in C++ for educational and research purposes.

CPlusTorch is a from-scratch implementation of deep learning framework functionality similar to PyTorch, built entirely in C++. This project aims to provide a clear understanding of how modern deep learning frameworks work under the hood, making it an excellent educational resource for learning both C++ and deep learning concepts.

## ✨ Features

### Core Tensor Operations
- **Multi-dimensional Tensor Class**: Full support for n-dimensional arrays with dynamic shapes
- **Automatic Differentiation**: Complete autograd system for gradient computation
- **Broadcasting**: Automatic tensor broadcasting for element-wise operations
- **Memory Management**: Efficient memory handling with smart pointers

### Neural Network Components
- **Modular Architecture**: Base `Module` class for building custom neural networks
- **Linear Layers**: Fully connected layers with configurable input/output dimensions
- **Loss Functions**: Mean Squared Error (MSE) loss with gradient support
- **Parameter Management**: Automatic parameter registration and gradient tracking

### Optimization
- **Adam Optimizer**: Adaptive moment estimation with bias correction
- **Gradient Clipping**: Built-in gradient clipping to prevent gradient explosion
- **Parameter Updates**: Automatic parameter update mechanism

### Weight Initialization
- **Kaiming Uniform**: Proper weight initialization for deep networks
- **Bias Initialization**: Configurable bias initialization

## 🚀 Quick Start

### Prerequisites
- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.10 or higher

### Building the Project

```bash
# Clone the repository
git clone https://github.com/Y0sem1te/cplustorch.git
cd cplustorch

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make  # or mingw32-make on Windows with MinGW
```

### Running the Demo

```bash
# Run the neural network training demo
./main
```

The demo trains a multi-layer perceptron to learn the function `y = sin(2πx) + 0.5x²`.

## 🎬 Training Demo

Watch the neural network training process in action:

![Training Demo](assets/training_demo.gif)

*The demo shows real-time training progress with loss values and R² scores as the model learns to approximate the target function.*

## 📁 Project Structure

```
cplustorch/
├── aten/                      # Core tensor library (similar to PyTorch's ATen)
│   ├── include/
│   │   ├── tensor.hpp         # Main tensor class
│   │   ├── tensor_iterator.hpp # Tensor iteration utilities
│   │   └── native/
│   │       └── ops.hpp        # Native operations
│   └── src/                   # Implementation files
├── torch/                     # High-level neural network API
│   ├── include/
│   │   ├── autograd/
│   │   │   └── engine.hpp     # Autograd engine
│   │   ├── nn/
│   │   │   ├── module.hpp     # Base module class
│   │   │   ├── init.hpp       # Weight initialization
│   │   │   └── modules/
│   │   │       ├── linear.hpp # Linear layer
│   │   │       └── mse_loss.hpp # MSE loss function
│   │   └── optim/
│   │       ├── optimizer.hpp  # Base optimizer
│   │       └── adam.hpp       # Adam optimizer
│   └── src/                   # Implementation files
├── main.cpp                   # Demo application
├── CMakeLists.txt            # Build configuration
└── README.md                 # This file
```

## 💡 Usage Example

Here's a simple example of how to create and train a neural network:

```cpp
#include "./aten/include/tensor.hpp"
#include "./torch/include/nn/modules/linear.hpp"
#include "./torch/include/nn/modules/mse_loss.hpp"
#include "./torch/include/optim/adam.hpp"

// Define a simple MLP
class MLP : public torch::nn::Module {
public:
    MLP(size_t input_size, size_t hidden_size, size_t output_size) 
        : fc1(std::make_shared<torch::nn::Linear>(input_size, hidden_size)),
          fc2(std::make_shared<torch::nn::Linear>(hidden_size, output_size)) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    std::shared_ptr<at::Tensor> forward(const std::shared_ptr<at::Tensor> input) override {
        auto x = fc1->forward(input);
        // Apply ReLU activation
        for (size_t i = 0; i < x->data.size(); ++i) {
            if (x->data[i] < 0) x->data[i] = 0;
        }
        return fc2->forward(x);
    }

private:
    std::shared_ptr<torch::nn::Linear> fc1;
    std::shared_ptr<torch::nn::Linear> fc2;
};

int main() {
    // Create model
    MLP model(1, 16, 1);
    
    // Create optimizer
    auto params = model.parameters();
    torch::optim::Adam optimizer(params, 0.0001);
    
    // Create loss function
    torch::nn::MSELoss criterion;
    
    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        // Forward pass
        auto output = model.forward(input);
        auto loss = criterion.forward(output, target);
        
        // Backward pass
        optimizer.zero_grad();
        loss->grad[0] = 1.0;
        loss->backward();
        
        // Update parameters
        optimizer.step();
    }
    
    return 0;
}
```

## 🎯 Current Capabilities

- ✅ Multi-dimensional tensor operations
- ✅ Automatic differentiation (autograd)
- ✅ Linear layers with bias
- ✅ MSE loss function
- ✅ Adam optimizer with gradient clipping
- ✅ Kaiming weight initialization
- ✅ Complete training pipeline
- ✅ Model parameter management

## 🚧 Roadmap

### Planned Features
- [ ] More activation functions (Sigmoid, Tanh, etc.)
- [ ] Additional loss functions (CrossEntropy, etc.)
- [ ] Convolutional layers
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] More optimizers (SGD, RMSprop)
- [ ] Learning rate scheduling
- [ ] Model serialization/deserialization
- [ ] GPU support (CUDA)

### Potential Enhancements
- [ ] SIMD optimizations
- [ ] Multi-threading support
- [ ] Memory pool allocation
- [ ] Advanced tensor operations
- [ ] Python bindings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Guidelines
1. Follow C++17 standards
2. Maintain consistent code style
3. Add appropriate comments and documentation
4. Test your changes thoroughly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by PyTorch framework architecture
- Built for educational purposes to understand deep learning internals
- Thanks to the open-source community for inspiration and best practices

## 📚 Educational Value

This project is particularly valuable for:
- **Students** learning C++ and deep learning concepts
- **Researchers** who want to understand framework internals
- **Developers** interested in implementing ML frameworks
- **Anyone** curious about how PyTorch works under the hood

---
