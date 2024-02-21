#ifndef YOLOV4_ANCHOR_H
#define YOLOV4_ANCHOR_H

#include <torch/torch.h>

class YOLOv4_AnchorImpl : public torch::nn::Module
{
public:
	explicit YOLOv4_AnchorImpl()
	{
		this->anchor_whs["small"] = std::vector<float>({ 10, 13, 16, 30, 33, 23 });
		this->anchor_whs["middle"] = std::vector<float>({ 30, 61, 62, 45, 59, 119 });
		this->anchor_whs["large"] = std::vector<float>({ 116, 90, 156, 198, 373, 326 });

		std::cout << "yolov4 anchor" << std::endl;
	}

	std::vector<torch::Tensor> create_anchors()
	{
		torch::Tensor wh_large = torch::from_blob(this->anchor_whs["large"].data(), { 3,2 }, torch::kFloat32) / 32.0f;
		torch::Tensor wh_middle = torch::from_blob(this->anchor_whs["middle"].data(), { 3, 2 }, torch::kFloat32) / 16.0f;
		torch::Tensor wh_small = torch::from_blob(this->anchor_whs["small"].data(), { 3, 2 }, torch::kFloat32) / 8.0f;

		torch::Tensor center_anchors_large = this->anchor_for_scale(13, wh_large);
		torch::Tensor center_anchors_middle = this->anchor_for_scale(26, wh_middle);
		torch::Tensor center_anchors_small = this->anchor_for_scale(52, wh_small);

		return std::vector<torch::Tensor>({ center_anchors_large, center_anchors_middle, center_anchors_small });
	}

	std::map<std::string, std::vector<float>> GetAnchor_whs()
	{
		return this->anchor_whs;
	}

private:
	std::map<std::string, std::vector<float>> anchor_whs;

	torch::Tensor anchor_for_scale(int grid_size, torch::Tensor wh)
	{
		std::vector<float> center_anchors;
		for (int y = 0; y < grid_size; ++y)
		{
			for (int x = 0; x < grid_size; ++x)
			{
				float cx = x + 0.5f;
				float cy = y + 0.5f;

				for (int i = 0; i < wh.sizes()[0]; ++i)
				{
					float w = wh[i][0].item<float>();
					float h = wh[i][1].item<float>();

					center_anchors.push_back(cx);
					center_anchors.push_back(cy);
					center_anchors.push_back(w);
					center_anchors.push_back(h);
				}
			}
		}

		return torch::from_blob(center_anchors.data(), { grid_size * grid_size * 3 * 4 }, torch::kFloat32).
			view({ grid_size, grid_size, 3, 4 }).clone();
	}

};

TORCH_MODULE(YOLOv4_Anchor);

#endif // YOLOV4_ANCHOR_H