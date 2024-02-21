//#ifndef UTILS_H
//#define UTILS_H
//
//#include <torch/torch.h>
//
//namespace dldetection {
//	torch::Tensor DropConnect(torch::Tensor inputs, float p, bool isTraining);
//
//	int RoundFilters(int filters, float widthCoefficient, int depthDivisor, int minDepth = -1);
//
//	template<typename SetT1, typename SetT2>
//	std::vector<std::tuple<SetT1, SetT2>> CartesianProduct(const std::vector<SetT1>& A, const std::vector<SetT2>& B)
//	{
//		std::vector<std::tuple<SetT1, SetT2>> product;
//		for (size_t i = 0; i < A.size(); ++i)
//		{
//			for (size_t j = 0; j < B.size(); ++j)
//			{
//				product.push_back(std::make_tuple(A[i], B[j]);
//			}
//		}
//
//		return product;
//	}
//
//	template<typename Base, typename T>
//	inline bool InstanceOf(const T*)
//	{
//		return std::is_base_of<Base, T>::value;
//	}
//}
//
//
//#endif