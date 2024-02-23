#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h> // @manual=//caffe2:torch_extension
#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

using namespace ::std;

vector<vector<uint32_t>> edit_distance2_with_dp(
    vector<uint32_t>& x,
    vector<uint32_t>& y) {
  uint32_t lx = x.size();
  uint32_t ly = y.size();
  vector<vector<uint32_t>> d(lx + 1, vector<uint32_t>(ly + 1));
  for (uint32_t i = 0; i < lx + 1; i++) {
    d[i][0] = i;
  }
  for (uint32_t j = 0; j < ly + 1; j++) {
    d[0][j] = j;
  }
  for (uint32_t i = 1; i < lx + 1; i++) {
    for (uint32_t j = 1; j < ly + 1; j++) {
      d[i][j] =
          min(min(d[i - 1][j], d[i][j - 1]) + 1,
              d[i - 1][j - 1] + 2 * (x.at(i - 1) == y.at(j - 1) ? 0 : 1));
    }
  }
  return d;
}
