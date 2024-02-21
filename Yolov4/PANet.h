#ifndef PANET_H
#define PANET_H


#include <torch/torch.h>
#include <vector>
#include "Mish.h"

class PANetImpl : public torch::nn::Module
{
public:
	explicit PANetImpl();
	std::vector<torch::Tensor> forward(torch::Tensor P5, torch::Tensor P4, torch::Tensor P3);

private:
	torch::nn::Sequential p52d5{ nullptr };
	torch::nn::Sequential p42p4_{ nullptr };
	torch::nn::Sequential p32p3_{ nullptr };
	torch::nn::Sequential d5_p4_2d4{ nullptr };
	torch::nn::Sequential d4_p3_2d3{ nullptr };
	torch::nn::Sequential d52d5_{ nullptr };
	torch::nn::Sequential d42d4_{ nullptr };
	torch::nn::Sequential u32u3_{ nullptr };
	torch::nn::Sequential u42u4_{ nullptr };
	torch::nn::Sequential d4u3_2u4{ nullptr };
	torch::nn::Sequential d5u4_2u5{ nullptr };
};


TORCH_MODULE(PANet);

#endif // PANET_H