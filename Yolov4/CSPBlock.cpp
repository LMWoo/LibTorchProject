#include "CSPBlock.h"

CSPBlockImpl::CSPBlockImpl(int in_channel, bool is_first, int num_blocks)
{
	if (is_first)
	{
		this->part1_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel),
			Mish()
		);

		this->part2_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel),
			Mish()
		);

		this->features = torch::nn::Sequential();
		for (int i = 0; i < num_blocks; ++i)
		{
			this->features->push_back(ResidualBlock(in_channel, in_channel / 2));
		}

		this->transition1_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel),
			Mish()
		);

		this->transition2_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(2 * in_channel, in_channel, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel),
			Mish()
		);
	}
	else
	{
		this->part1_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel / 2, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel / 2),
			Mish()
		);

		this->part2_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel / 2, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel / 2),
			Mish()
		);

		this->features = torch::nn::Sequential();
		for (int i = 0; i < num_blocks; ++i)
		{
			this->features->push_back(ResidualBlock(in_channel / 2));
		}

		this->transition1_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel / 2, in_channel / 2, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel / 2),
			Mish()
		);

		this->transition2_conv = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel, 1).stride(1).padding(0).bias(false)),
			torch::nn::BatchNorm2d(in_channel),
			Mish()
		);
	}

	register_module("part1_conv", this->part1_conv);
	register_module("part2_conv", this->part2_conv);
	register_module("features", this->features);
	register_module("transition1_conv", this->transition1_conv);
	register_module("transition2_conv", this->transition2_conv);
}

torch::Tensor CSPBlockImpl::forward(torch::Tensor x)
{
	torch::Tensor part1 = this->part1_conv->forward(x);
	torch::Tensor part2 = this->part2_conv->forward(x);

	torch::Tensor residual = part2.clone();
	part2 = this->features->forward(part2);
	part2 += residual;
	part2 = this->transition1_conv->forward(part2);

	x = this->transition2_conv->forward(torch::cat({ part1, part2 }, 1));
	return x;
}