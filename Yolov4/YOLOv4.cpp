#include "YOLOv4.h"

YOLOv4Impl::YOLOv4Impl(CSPDarknet53 backbone, int num_classes)
{
	this->num_classes = num_classes;
	this->backbone = backbone;
	this->SPP = SPPNet();
	this->PANET = PANet();

	this->pred_s = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 3 * (1 + 4 + this->num_classes), 1).stride(1).padding(0).bias(true))
	);

	this->pred_m = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(512),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 3 * (1 + 4 + this->num_classes), 1).stride(1).padding(0).bias(true))
	);

	this->pred_l = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(1024),
		Mish(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 3 * (1 + 4 + this->num_classes), 1).stride(1).padding(0).bias(true))
	);

	register_module("backbone", this->backbone);
	register_module("SPP", this->SPP);
	register_module("PANET", this->PANET);
	register_module("pred_s", this->pred_s);
	register_module("pred_m", this->pred_m);
	register_module("pred_l", this->pred_l);
}

std::vector<torch::Tensor> YOLOv4Impl::forward(torch::Tensor x)
{
	torch::Tensor P3 = this->backbone->features1->forward(x);
	torch::Tensor P4 = this->backbone->features2->forward(P3);
	torch::Tensor P5 = this->backbone->features3->forward(P4);


	P5 = this->SPP->forward(P5);
	std::vector<torch::Tensor> U = this->PANET->forward(P5, P4, P3);

	torch::Tensor p_l = this->pred_l->forward(U[0]).permute({ 0, 2, 3, 1 });
	torch::Tensor p_m = this->pred_m->forward(U[1]).permute({ 0, 2, 3, 1 });
	torch::Tensor p_s = this->pred_s->forward(U[2]).permute({ 0, 2, 3, 1 });

	return std::vector<torch::Tensor>({ p_l, p_m, p_s });
}