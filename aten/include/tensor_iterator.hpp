#pragma once

#include <functional>
#include <vector>
#include <stdexcept>

namespace at {
class Tensor;
using i64 = long long;
std::vector<size_t> broadcastShape(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2);
std::vector<size_t> transferIdx(const std::vector<size_t> &idx, const std::vector<size_t> &org_shape);
void traverse(const std::vector<size_t> &shape, std::function<void(const std::vector<size_t> &idx)> func);

} // namespace at