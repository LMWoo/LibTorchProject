#ifndef YOLOV4_CODER_H
#define YOLOV4_CODER_H

#include <torch/torch.h>
#include "YOLOv4_Anchor.h"
#include "util.h"

class YOLOv4_CoderImpl : public torch::nn::Module
{
public:
	explicit YOLOv4_CoderImpl(torch::Device device_)
		: device(device_)
	{
		this->anchor_whs = this->anchor->GetAnchor_whs();
		std::vector<torch::Tensor> center_anchors = this->anchor->create_anchors();
		this->center_anchor_l = center_anchors[0];
		this->center_anchor_m = center_anchors[1];
		this->center_anchor_s = center_anchors[2];

		this->num_classes = 20;
		this->assign_anchors_to_device(device_);
	}

	void assign_anchors_to_device(torch::Device device)
	{
		this->center_anchor_l = this->center_anchor_l.to(device);
		this->center_anchor_m = this->center_anchor_m.to(device);
		this->center_anchor_s = this->center_anchor_s.to(device);
	}

	int GetNumClasses()
	{
		return this->num_classes;
	}

	std::vector<torch::Tensor> build_target(std::vector<torch::Tensor> gt_boxes, std::vector<torch::Tensor> gt_labels, bool IT = false);
	std::vector<torch::Tensor> split_preds(torch::Tensor pred);
	std::vector<torch::Tensor> decode(std::vector<torch::Tensor> gcxgcys);

private:
	YOLOv4_Anchor anchor;
	std::map<std::string, std::vector<float>> anchor_whs;
	torch::Tensor center_anchor_l;
	torch::Tensor center_anchor_m;
	torch::Tensor center_anchor_s;
	torch::Device device;
	int num_classes;
};

TORCH_MODULE(YOLOv4_Coder);

#endif // YOLOV4_CODER_H