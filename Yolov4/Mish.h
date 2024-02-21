#ifndef MISH_H
#define MISH_H

#include <torch/torch.h>

class MishImpl : public torch::nn::Module
{
public:
	explicit MishImpl();
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Mish);

#endif // MISH_H