#include "CSPDarknet53.h"

CSPDarknet53Impl::CSPDarknet53Impl(int num_classes, bool pretrained)
{
	this->num_classes = num_classes;

	this->features1 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(32),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(64),
		Mish(),
		CSPBlock(64, true, 1),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(128),
		Mish(),
		CSPBlock(128, false, 2),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		CSPBlock(256, false, 8)
	);

	this->features2 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		CSPBlock(512, false, 8)
	);

	this->features3 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(1024),
		Mish(),
		CSPBlock(1024, false, 4)
	);

	this->gap = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
	this->fc = torch::nn::Linear(torch::nn::LinearOptions(1024, this->num_classes));

	register_module("features1", this->features1);
	register_module("features2", this->features2);
	register_module("features3", this->features3);
	register_module("gap", this->gap);
	register_module("fc", this->fc);
}

torch::Tensor CSPDarknet53Impl::forward(torch::Tensor x)
{
	x = this->features1->forward(x);
	x = this->features2->forward(x);
	x = this->features3->forward(x);
	x = this->gap->forward(x);
	x = x.view({ -1, 1024 });
	x = this->fc->forward(x);
	return x;
}
