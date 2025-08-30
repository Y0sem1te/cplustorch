#pragma once

#include "optimizer.hpp"
#include <vector>
#include <memory>
#include <cstddef>

namespace at { class Tensor; }

namespace torch::optim {

class Adam : public Optimizer {
public:
	Adam(const std::vector<std::shared_ptr<at::Tensor>>& params,
		 double lr = 1e-3,
		 double beta1 = 0.9,
		 double beta2 = 0.999,
		 double eps = 1e-8);

	void step() override;
	void zero_grad() override;

private:
	struct State {
		std::vector<double> m;
		std::vector<double> v;
	};

	std::vector<std::shared_ptr<at::Tensor>> params_;
	std::vector<State> states_;

	double lr_;
	double beta1_;
	double beta2_;
	double eps_;
	long long step_ = 0;
};

} // namespace torch::optim
