#ifndef YOLOV4_H
#define YOLOV4_H

#include <torch/torch.h>
#include "CSPDarknet53.h"
#include "SPPNet.h"
#include "PANet.h"
#include "Mish.h"

class YOLOv4Impl : public torch::nn::Module
{
public:
	explicit YOLOv4Impl(CSPDarknet53 backbone, int num_classes = 80);
	std::vector<torch::Tensor> forward(torch::Tensor x);

private:
	int num_classes;
	CSPDarknet53 backbone{ nullptr };
	SPPNet SPP{ nullptr };
	PANet PANET{ nullptr };

	torch::nn::Sequential pred_s{ nullptr };
	torch::nn::Sequential pred_m{ nullptr };
	torch::nn::Sequential pred_l{ nullptr };
};

TORCH_MODULE(YOLOv4);

#endif