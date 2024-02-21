#include "VOCDataset.h"

std::vector<torch::Tensor> VOCDataset::parse_voc(std::string xml_file_path)
{
	tinyxml2::XMLDocument doc;
	if (doc.LoadFile(xml_file_path.c_str()) != 0)
	{
		std::cout << "Not load xml : " + xml_file_path << std::endl;
		return std::vector<torch::Tensor>({ torch::randn({1}), torch::randn(1)});
	}
	
	tinyxml2::XMLElement* root = doc.RootElement();

	if (root == NULL)
	{
		std::cout << "Failed to load file : No root element." << std::endl;
		doc.Clear();
		return std::vector<torch::Tensor>({ torch::randn({1}), torch::randn(1)});
	}

	std::vector<float> vboxes;
	std::vector<float> vlabels;

	const char* elementValue = root->Value();

	for (tinyxml2::XMLElement* elem = root->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
	{
		std::string elemName = elem->Value();
		std::string name = "";

		if (strcmp(elemName.data(), "object") == 0)
		{
			for (tinyxml2::XMLNode* object = elem->FirstChildElement(); object != NULL; object = object->NextSiblingElement())
			{
				if (strcmp(object->Value(), "name") == 0)
				{
					name = object->FirstChild()->Value();
					vlabels.push_back(float(this->class_idx_dict[name]));
				}

				if (strcmp(object->Value(), "bndbox") == 0)
				{
					tinyxml2::XMLElement* xmin_ = object->FirstChildElement("xmin");
					tinyxml2::XMLElement* ymin_ = object->FirstChildElement("ymin");
					tinyxml2::XMLElement* xmax_ = object->FirstChildElement("xmax");
					tinyxml2::XMLElement* ymax_ = object->FirstChildElement("ymax");
					
					int xmin = std::atoi(std::string(xmin_->FirstChild()->Value()).c_str());
					int xmax = std::atoi(std::string(ymin_->FirstChild()->Value()).c_str());
					int ymin = std::atoi(std::string(xmax_->FirstChild()->Value()).c_str());
					int ymax = std::atoi(std::string(ymax_->FirstChild()->Value()).c_str());

					vboxes.push_back(float(xmin) - 1.0);
					vboxes.push_back(float(xmax) - 1.0);
					vboxes.push_back(float(ymin) - 1.0);
					vboxes.push_back(float(ymax) - 1.0);
				}
			}
		}
	}

	torch::Tensor boxes = torch::from_blob(vboxes.data(), { int(vboxes.size()) }, torch::kFloat32).view({ -1, 4 });
	torch::Tensor labels = torch::from_blob(vlabels.data(), { int(vlabels.size()) }, torch::kFloat32);

	doc.Clear();
	return std::vector<torch::Tensor>({ boxes.clone(), labels.clone()});
}

torch::data::Example<> VOCDataset::get(size_t idx)
{
	std::string image_path = this->img_list[idx];

	cv::Mat image = cv::imread(this->img_list[idx]);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	std::vector<torch::Tensor> voc_outputs = this->parse_voc(this->anno_list[idx]);
	torch::Tensor boxes = voc_outputs[0];
	torch::Tensor labels = voc_outputs[1];
	std::vector<std::string> transform_list({ "resize" });
	
	bool zero_to_one_coord = true;
	std::vector<torch::Tensor> transform_outputs = transform::transform(image, boxes, labels, this->split, transform_list, this->resize, zero_to_one_coord);

	torch::Tensor out_image = transform_outputs[0];
	torch::Tensor out_boxes = transform_outputs[1];
	torch::Tensor out_labels = transform_outputs[2];

	torch::Tensor out_target = torch::concat({ out_boxes, out_labels.unsqueeze(-1)}, 1);
	return { out_image, out_target };
}