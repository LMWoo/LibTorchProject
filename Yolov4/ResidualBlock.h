#ifndef RESIDUALBLOCK_H
#define RESIDUALBLOCK_H

#include <torch/torch.h>
#include "Mish.h"

class ResidualBlockImpl : public torch::nn::Module
{
public:
	explicit ResidualBlockImpl(int in_channel, int hidden_channel = -1);
	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::Sequential features;
};

TORCH_MODULE(ResidualBlock);

#endif // RESIDUALBLOCK_H