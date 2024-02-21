#include "SPPNet.h"

SPPNetImpl::SPPNetImpl()
{
	this->conv1 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish()
	);
	
	this->conv2 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(2048, 2048, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(2048),
		Mish()
	);

	this->maxpool5 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(5).kernel_size(5).stride(1).padding(5 / 2));
	this->maxpool9 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(9).kernel_size(9).stride(1).padding(9 / 2));
	this->maxpool13 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(13).kernel_size(13).stride(1).padding(13 / 2));

	register_module("conv1", this->conv1);
	register_module("conv2", this->conv2);
	register_module("maxpool5", this->maxpool5);
	register_module("maxpool9", this->maxpool9);
	register_module("maxpool13", this->maxpool13);
}

torch::Tensor SPPNetImpl::forward(torch::Tensor x)
{
	x = this->conv1->forward(x);
	torch::Tensor maxpool5 = this->maxpool5->forward(x);
	torch::Tensor maxpool9 = this->maxpool9->forward(x);
	torch::Tensor maxpool13 = this->maxpool13->forward(x);

	x = torch::cat({ x, maxpool5, maxpool9, maxpool13 }, 1);
	x = this->conv2->forward(x);
	return x;
}