#include "ResidualBlock.h"

ResidualBlockImpl::ResidualBlockImpl(int in_channel, int hidden_channel)
{
	if (hidden_channel == -1)
		hidden_channel = in_channel;

	this->features = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, hidden_channel, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(hidden_channel),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_channel, in_channel, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(in_channel),
		Mish()
	);
	
	register_module("features", this->features);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x)
{
	torch::Tensor residual = x.clone();
	x = this->features->forward(x);
	x += residual;
	return x;
}