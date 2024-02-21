#ifndef UTIL_H
#define UTIL_H

#include <torch/torch.h>

namespace util
{
	torch::Tensor cxcy_to_xy(torch::Tensor cxcy);
	torch::Tensor xy_to_cxcy(torch::Tensor xy);
	torch::Tensor find_jaccard_overlap(torch::Tensor set_1, torch::Tensor set_2, float eps);
	torch::Tensor find_intersection(torch::Tensor set_1, torch::Tensor set_2);
}

#endif // UTIL_H