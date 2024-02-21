#include "transform.h"

torch::Tensor transform::to_tensor(cv::Mat image)
{
	int c = image.channels();
	int width = image.size[1];
	int height = image.size[0];

	image.convertTo(image, CV_32FC3, 1.0 / 255.0f);

	torch::Tensor output = torch::from_blob(image.data, { width, height, c });

	return output.permute({ 2, 0, 1 }).clone();
}

std::tuple<cv::Mat, torch::Tensor> transform::resize(cv::Mat image, torch::Tensor boxes, int new_size, bool zero_to_one_coord)
{
	cv::Mat new_image;
	cv::resize(image, new_image, cv::Size(new_size, new_size));


	std::vector<float> vdims({ float(image.size[1]), float(image.size[0]), float(image.size[1]), float(image.size[0])});
	torch::Tensor old_dims = torch::from_blob(vdims.data(), { int64_t(vdims.size()) }, torch::kFloat32).unsqueeze(0);
	
	torch::Tensor new_boxes = boxes / old_dims;

	if (zero_to_one_coord == false)
	{
		// torch::Tensor new_dims = torch::dims
	}

	return std::tuple<cv::Mat, torch::Tensor>({ new_image, new_boxes});
}

std::vector<torch::Tensor> transform::transform(cv::Mat image, torch::Tensor boxes, torch::Tensor labels, std::string split, std::vector<std::string> transform_list, int new_size, bool zero_to_one_coord)
{
	if (split.compare("train") == 0)
	{

	}

	cv::Mat new_image = image.clone();
	torch::Tensor new_boxes = boxes.clone();
	torch::Tensor new_labels = labels.clone();

	auto it = std::find(transform_list.begin(), transform_list.end(), "resize");
	if (it != transform_list.end())
	{
		std::tuple<cv::Mat, torch::Tensor> outputs = resize(new_image, new_boxes, new_size, zero_to_one_coord);
		new_image = std::get<0>(outputs);
		new_boxes = std::get<1>(outputs);
	}

	torch::Tensor out_image = to_tensor(new_image);

	out_image[0] = out_image[0].sub_(0.485).div_(0.229);
	out_image[1] = out_image[1].sub_(0.456).div_(0.224);
	out_image[2] = out_image[2].sub_(0.406).div_(0.225);

	return std::vector<torch::Tensor>({ out_image, new_boxes, new_labels});
}