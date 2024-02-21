#ifndef SPPNET_H
#define SPPNET_H

#include <torch/torch.h>
#include "Mish.h"

class SPPNetImpl : public torch::nn::Module
{
public:
	explicit SPPNetImpl();
	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Sequential conv1{ nullptr };
	torch::nn::Sequential conv2{ nullptr };

	torch::nn::MaxPool2d maxpool5{ nullptr };
	torch::nn::MaxPool2d maxpool9{ nullptr };
	torch::nn::MaxPool2d maxpool13{ nullptr };
};

TORCH_MODULE(SPPNet);

#endif // SPPNET_H