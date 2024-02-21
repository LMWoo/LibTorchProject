#include "Mish.h"

MishImpl::MishImpl()
{
}

torch::Tensor MishImpl::forward(torch::Tensor x) {
	return x * torch::tanh(torch::softplus(x));
}