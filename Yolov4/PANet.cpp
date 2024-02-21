#include "PANet.h"

PANetImpl::PANetImpl()
{
	this->p52d5 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(2048, 512, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(1024),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish()
	);
	

	this->p42p4_ = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish()
	);

	
	this->p32p3_ = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(128),
		Mish()
	);

	this->d5_p4_2d4 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish()
	);

	this->d4_p3_2d3 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(128),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(128),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(128),
		Mish()
	);

	this->d52d5_ = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kNearest))
	);

	this->d42d4_ = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(128),
		Mish(),
		torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kNearest))
	);

	this->u32u3_ = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish()
	);


	this->u42u4_ = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish()
	);

	
	this->d4u3_2u4 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish()
	);


	this->d5u4_2u5 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(1024),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(1024),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1).padding(0).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish()
	);

	register_module("p52d5", this->p52d5);
	register_module("p42p4_", this->p42p4_);
	register_module("p32p3_", this->p32p3_);
	register_module("d5_p4_2d4", this->d5_p4_2d4);
	register_module("d4_p3_2d3", this->d4_p3_2d3);
	register_module("d52d5_", this->d52d5_);
	register_module("d42d4_", this->d42d4_);
	register_module("u32u3_", this->u32u3_);
	register_module("u42u4_", this->u42u4_);
	register_module("d4u3_2u4", this->d4u3_2u4);
	register_module("d5u4_2u5", this->d5u4_2u5);
}

std::vector<torch::Tensor> PANetImpl::forward(torch::Tensor P5, torch::Tensor P4, torch::Tensor P3)
{
	torch::Tensor D5 = this->p52d5->forward(P5);
	torch::Tensor D5_ = this->d52d5_->forward(D5);
	torch::Tensor P4_ = this->p42p4_->forward(P4);
	torch::Tensor D4 = this->d5_p4_2d4->forward(torch::cat({ D5_, P4_ }, 1));
	torch::Tensor D4_ = this->d42d4_->forward(D4);
	torch::Tensor P3_ = this->p32p3_->forward(P3);
	torch::Tensor D3 = this->d4_p3_2d3->forward(torch::cat({ D4_, P3_ }, 1));

	torch::Tensor U3 = D3.clone();
	torch::Tensor U3_ = this->u32u3_->forward(U3);
	torch::Tensor U4 = this->d4u3_2u4->forward(torch::cat({ D4, U3_ }, 1));
	torch::Tensor U4_ = this->u42u4_->forward(U4);
	torch::Tensor U5 = this->d5u4_2u5->forward(torch::cat({ D5, U4_ }, 1));

	return std::vector<torch::Tensor>({ U5, U4, U3 });
}