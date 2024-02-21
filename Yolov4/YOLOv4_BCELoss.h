#ifndef YOLOV4_BCELOSS_H
#define YOLOV4_BCELOSS_H

#include <torch/torch.h>

class YOLOv4_BCELossImpl : public torch::nn::Module
{
public:
	explicit YOLOv4_BCELossImpl() {}
	torch::Tensor forward(torch::Tensor pred, torch::Tensor target) {
		pred = clip_by_tensor(pred, 1e-7, 1.0 - 1e-7);
		auto output = -target * torch::log(pred) - (1.0 - target) * torch::log(1.0 - pred);
		return output;
	}

private:
	torch::Tensor clip_by_tensor(torch::Tensor t, float t_min, float t_max) {
		t = t.to(torch::kFloat32);
		auto result = (t >= t_min).to(torch::kFloat32) * t + (t < t_min).to(torch::kFloat32) * t_min;
		result = (result <= t_max).to(torch::kFloat32) * result + (result > t_max).to(torch::kFloat32) * t_max;
		return result;
	}
};

TORCH_MODULE(YOLOv4_BCELoss);

#endif // YOLOV4_BCELOSS_H