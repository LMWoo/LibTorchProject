#ifndef CSPBLOCK_H
#define CSPBLOCK_H

#include <torch/torch.h>
#include "Mish.h"
#include "ResidualBlock.h"

class CSPBlockImpl : public torch::nn::Module
{
public:
	explicit CSPBlockImpl(int in_channel, bool is_first=false, int num_blocks=1);
	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::Sequential part1_conv{ nullptr };
	torch::nn::Sequential part2_conv{ nullptr };
	torch::nn::Sequential features{ nullptr };
	torch::nn::Sequential transition1_conv{ nullptr };
	torch::nn::Sequential transition2_conv{ nullptr };
};

TORCH_MODULE(CSPBlock);

#endif // CSPBLOCK_H