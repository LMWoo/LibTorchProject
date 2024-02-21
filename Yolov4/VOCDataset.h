#ifndef VOCDATASET_H
#define VOCDATASET_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <sys/stat.h>
#include "tinyxml2.h"
#include "transform.h"

namespace fs = std::filesystem;

class VOCDataset : public torch::data::Dataset<VOCDataset>
{
public:
	VOCDataset(std::string root_, std::string split_, int resize_)
		: class_names({ "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
			"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" })
	{
		this->root = root_;
		this->split = split_;
		this->resize = resize_;

		fs::directory_iterator img_itr(root_ + "/" + split_ + "/images");

		while (img_itr != fs::end(img_itr))
		{
			const fs::directory_entry& entry = *img_itr;
			std::string img_file_ext = entry.path().extension().string();
			std::string img_file_name = entry.path().filename().string();

			img_file_name.erase(img_file_name.length() - img_file_ext.length(), img_file_name.length());

			std::string label_path = root_ + "/" + split_ + "/labels/" + img_file_name + ".xml";

			this->img_list.push_back(entry.path().string());
			this->anno_list.push_back(label_path);
			img_itr++;
		}

		std::cout << split_ << " data size : " << img_list.size() << std::endl;

		for (int i = 0; i < this->class_names.size(); ++i)
		{
			this->class_idx_dict[class_names[i]] = i;
			this->idx_class_dict[i] = class_names[i];
		}
	}

	torch::data::Example<> get(size_t idx) override;
	std::vector<ExampleType> get_batch(c10::ArrayRef<size_t> indices) override {
		std::vector<ExampleType> batch;
		batch.reserve(indices.size());
		for (const auto i : indices) {
			batch.push_back(get(i));
		}
		return batch;
	}

	torch::optional<size_t> size() const override {
		return img_list.size();
	}

public:
	std::vector<std::string> class_names;

private:
	std::string root;
	std::string split;
	int resize;

	std::vector<std::string> img_list;
	std::vector<std::string> anno_list;

	std::map<std::string, int> class_idx_dict;
	std::map<int, std::string> idx_class_dict;

private:
	std::vector<torch::Tensor> parse_voc(std::string xml_file_path);
};


#endif // VOCDATASET_H