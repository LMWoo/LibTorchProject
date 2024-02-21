#ifndef YOLOV4_LOSS_H
#define YOLOV4_LOSS_H

#include <torch/torch.h>
#include "YOLOv4_Coder.h"
#include "YOLOv4_MSELoss.h"
#include "YOLOv4_BCELoss.h"
#include "util.h"

class YOLOv4_LossImpl : public torch::nn::Module
{
public:
	explicit YOLOv4_LossImpl(YOLOv4_Coder coder)
	{
		this->coder = coder;
		this->mse = YOLOv4_MSELoss();
		this->bce = YOLOv4_BCELoss();
		this->num_classes = this->coder->GetNumClasses();
		
		std::cout << "LOSS num_classes : " << this->num_classes << std::endl;
	}

	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> pred, std::vector<torch::Tensor> gt_boxes, std::vector<torch::Tensor> gt_labels);

private:
	torch::Tensor giou_loss(torch::Tensor boxes1, torch::Tensor boxes2);

private:
	YOLOv4_Coder coder{ nullptr };
	YOLOv4_MSELoss mse{ nullptr };
	YOLOv4_BCELoss bce{ nullptr };
	int num_classes;
};

TORCH_MODULE(YOLOv4_Loss);

#endif // YOLOV4_LOSS_H