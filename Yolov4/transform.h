#ifndef TRANSFORM_H
#define TRANSFORM_H


#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace transform
{
	torch::Tensor to_tensor(cv::Mat image);
	std::tuple<cv::Mat, torch::Tensor> resize(cv::Mat image, torch::Tensor boxes, int new_size, bool zero_to_one_coord);
	std::vector<torch::Tensor> transform(cv::Mat image, torch::Tensor boxes, torch::Tensor labels, std::string split, std::vector<std::string> transform_list, int new_size, bool zero_to_one_coord);
}

#endif // TRANSFORM_H