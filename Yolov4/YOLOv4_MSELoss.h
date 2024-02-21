#ifndef YOLOV4_MSELOSS_H
#define YOLOV4_MSELOSS_H

#include <torch/torch.h>

class YOLOv4_MSELossImpl : public torch::nn::Module
{
public:
	explicit YOLOv4_MSELossImpl() {}

	torch::Tensor forward(torch::Tensor pred, torch::Tensor target) {
		return torch::pow((pred - target), 2);
	}
};

TORCH_MODULE(YOLOv4_MSELoss);

#endif // YOLOV4_MSELOSS_H