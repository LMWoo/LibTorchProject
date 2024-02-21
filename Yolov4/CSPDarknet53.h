#ifndef CSPDARKNET53_H
#define CSPDARKNET53_H

#include <torch/torch.h>
#include "CSPBlock.h"
#include "Mish.h"

class CSPDarknet53Impl : public torch::nn::Module
{
public:
	explicit CSPDarknet53Impl(int num_classes = 1000, bool pretrained = true);
	torch::Tensor forward(torch::Tensor x);
	//void load_darknet_weights(const char* pretrained_file);
	//void count_parameters();

public:
	torch::nn::Sequential features1{ nullptr };
	torch::nn::Sequential features2{ nullptr };
	torch::nn::Sequential features3{ nullptr };

private:
	int num_classes;
	
	torch::nn::AdaptiveAvgPool2d gap{ nullptr };
	torch::nn::Linear fc{ nullptr };

	// void init_layer();
};

TORCH_MODULE(CSPDarknet53);

#endif // CSPDARKNET53_H