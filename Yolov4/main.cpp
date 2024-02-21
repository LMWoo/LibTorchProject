#pragma once
#include <torch/torch.h>
#include <vector>
#include "YOLOv4_Loss.h"
#include "YOLOv4.h"
#include "VOCDataset.h"
#include <Windows.h>

int main()
{
	double lr = 1e-3;
	double momentum = 0.9;
	int num_classes = 20;
	int batch_size = 2;

	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available())
	{
		device = torch::Device(torch::kCUDA);
	}
	
	VOCDataset train_set = VOCDataset("C:/Users/minwoo/Desktop/Projects/Yolov4/Resources/dataset", "train", 416);
	VOCDataset test_set = VOCDataset("C:/Users/minwoo/Desktop/Projects/Yolov4/Resources/dataset", "test", 416);
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), batch_size);
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_set), 1);

	YOLOv4 model = YOLOv4(CSPDarknet53(), num_classes);
	model->to(device);
	
	YOLOv4_Coder yolo4_coder = YOLOv4_Coder(device);
	YOLOv4_Loss criterion = YOLOv4_Loss(yolo4_coder);
	criterion->to(device);

	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr).momentum(momentum));

	LARGE_INTEGER	liFrequency;
	QueryPerformanceFrequency(&liFrequency);
	for (int epoch = 0; epoch < 200; ++epoch)
	{
		std::cout << "Training of epoch " << epoch << std::endl;
		model->train();

		int print_step = 100;
		int current_step = 0;
		LARGE_INTEGER liStartTime, liEndTime;
		QueryPerformanceCounter(&liStartTime);
		for (auto& batch : *train_loader)
		{
			std::vector<torch::Tensor> images_vec = {};
			std::vector<torch::Tensor> boxes_vec = {};
			std::vector<torch::Tensor> labels_vec = {};

			for (int i = 0; i < batch_size; ++i)
			{
				images_vec.push_back(batch[i].data.to(device));
				boxes_vec.push_back(batch[i].target.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 4)}).to(device));
				labels_vec.push_back(batch[i].target.index({ torch::indexing::Ellipsis, 4 }).to(device));
			}
			torch::Tensor data = torch::stack(images_vec);

			std::vector<torch::Tensor> outputs = model->forward(data);

			std::vector<torch::Tensor> loss = criterion->forward(outputs, boxes_vec, labels_vec);

			if (current_step != 0 && current_step % print_step == 0)
			{
				QueryPerformanceCounter(&liEndTime);
				__int64 nTimeMS = (liEndTime.QuadPart - liStartTime.QuadPart) / (liFrequency.QuadPart / 1000);

				std::cout << nTimeMS << std::endl;
				std::cout << "Loss : " << loss[0] << std::endl;
				QueryPerformanceCounter(&liStartTime);
			}

			optimizer.zero_grad();
			loss[0].backward();
			optimizer.step();
			current_step++;
		}

	}

	return 0;
}

//int main()
//{
//	torch::Device device(torch::kCPU);
//	YOLOv4 model = YOLOv4(CSPDarknet53(1000, true));
//	model->to(device);
//
//	std::vector<torch::Tensor> p = model->forward(torch::randn({ 1, 3, 416, 416 }).to(device));
//	std::cout << "large : " << p[0].sizes() << std::endl;
//	std::cout << "medium : " << p[1].sizes() << std::endl;
//	std::cout << "small : " << p[2].sizes() << std::endl;
//
//	torch::Tensor x = torch::randn({ 4, 3 });
//	int arr[] = {0, 3, 2, 1};
//	torch::Tensor y = torch::from_blob(arr, { 4 }, torch::kFloat32);
//	std::cout << x << std::endl;
//	y.toType(torch::kInt);
//	std::cout << y << std::endl;
//	std::cout << x.index_select(0, y) << std::endl;
// 
//	return 0;
//}

// Dataset
//#include <torch/torch.h>
//#include <opencv2/opencv.hpp>
//#include <filesystem>
//#include <sys/stat.h>
//#include "tinyxml2.h"
//
//namespace fs = std::filesystem;
//
//struct BBox
//{
//	int xmin = 0;
//	int xmax = 0;
//	int ymin = 0;
//	int ymax = 0;
//	std::string name = "";
//	int GetH();
//	int GetW();
//	float CenterX();
//	float CenterY();
//};
//
//int BBox::GetH()
//{
//	return ymax - ymin;
//}
//
//int BBox::GetW()
//{
//	return xmax - xmin;
//}
//
//float BBox::CenterX()
//{
//	return (xmax + xmin) / 2.0;
//}
//
//float BBox::CenterY()
//{
//	return  (ymax + ymin) / 2.0;
//}
//
//struct Data {
//	Data(cv::Mat img, std::vector<BBox> boxes) : image(img), bboxes(boxes) {};
//	cv::Mat image;
//	std::vector<BBox> bboxes;
//};
//
//class Augmentations
//{
//public:
//	static Data Resize(Data mData, int width, int height, float probability);
//};
//
//template<typename T>
//T RandomNum(T _min, T _max)
//{
//	T temp;
//	if (_min > _max)
//	{
//		temp = _max;
//		_max = _min;
//		_min = temp;
//	}
//	return rand() / (double)RAND_MAX * (_max - _min) + _min;
//}
//
//Data Augmentations::Resize(Data mData, int width, int height, float probability)
//{
//	float rand_number = RandomNum<float>(0, 1);
//	if (rand_number <= probability)
//	{
//		float h_scale = height * 1.0 / mData.image.rows;
//		float w_scale = width * 1.0 / mData.image.cols;
//		for (int i = 0; i < mData.bboxes.size(); ++i)
//		{
//			mData.bboxes[i].xmin = int(w_scale * mData.bboxes[i].xmin);
//			mData.bboxes[i].xmax = int(w_scale * mData.bboxes[i].xmax);
//			mData.bboxes[i].ymin = int(h_scale * mData.bboxes[i].ymin);
//			mData.bboxes[i].ymax = int(h_scale * mData.bboxes[i].ymax);
//		}
//
//		cv::resize(mData.image, mData.image, cv::Size(width, height));
//	}
//	return mData;
//}
//
//std::vector<BBox> loadXML(std::string xml_path)
//{
//	std::vector<BBox> objects;
//
//	tinyxml2::XMLDocument doc;
//	if (doc.LoadFile(xml_path.c_str()) != 0)
//	{
//		std::cerr << "Not load xml" << std::endl;
//		return objects;
//	}
//
//	tinyxml2::XMLElement* root = doc.RootElement();
//
//	if (root == NULL)
//	{
//		std::cerr << "Failed to load file : No root element." << std::endl;
//
//		doc.Clear();
//		return objects;
//	}
//
//	const char* elementValue = root->Value();
//
//	for (tinyxml2::XMLElement* elem = root->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
//	{
//		std::string elemName = elem->Value();
//		std::string name = "";
//
//		if (strcmp(elemName.data(), "object") == 0)
//		{
//			for (tinyxml2::XMLNode* object = elem->FirstChildElement(); object != NULL; object = object->NextSiblingElement())
//			{
//				if (strcmp(object->Value(), "name") == 0)
//				{
//					name = object->FirstChild()->Value();
//				}
//
//				if (strcmp(object->Value(), "bndbox") == 0)
//				{
//					BBox obj;
//					tinyxml2::XMLElement* xmin = object->FirstChildElement("xmin");
//					tinyxml2::XMLElement* ymin = object->FirstChildElement("ymin");
//					tinyxml2::XMLElement* xmax = object->FirstChildElement("xmax");
//					tinyxml2::XMLElement* ymax = object->FirstChildElement("ymax");
//
//					obj.xmin = atoi(std::string(xmin->FirstChild()->Value()).c_str());
//					obj.xmax = atoi(std::string(xmax->FirstChild()->Value()).c_str());
//					obj.ymin = atoi(std::string(ymin->FirstChild()->Value()).c_str());
//					obj.ymax = atoi(std::string(ymax->FirstChild()->Value()).c_str());
//					obj.name = name;
//					objects.push_back(obj);
//				}
//			}
//		}
//	}
//
//	doc.Clear();
//	return objects;
//}
//
//void load_det_data_from_folder(std::string label_folder, std::string image_folder, std::string image_type, std::vector<std::string>& list_images, std::vector<std::string>& list_labels)
//{
//	fs::directory_iterator itr(label_folder);
//
//	while (itr != fs::end(itr))
//	{
//		const fs::directory_entry& entry = *itr;
//
//		list_labels.push_back(entry.path().string());
//		std::string image_file_ext = entry.path().extension().string();
//		std::string image_file_name = entry.path().filename().string();
//
//		image_file_name.erase(image_file_name.length() - image_file_ext.length(), image_file_name.length());
//
//		std::string image_path = image_folder + "/" + image_file_name + image_type;
//		list_images.push_back(image_path);
//
//		itr++;
//	}
//}
//
//class DetDataset : public torch::data::Dataset<DetDataset> {
//private:
//	std::vector<std::string> list_images;
//	std::vector<std::string> list_labels;
//	bool isTrain = true;
//	int width = 416; int height = 416;
//	float hflipProb = 0; float vflipProb = 0; float noiseProb = 0; float brightProb = 0;
//	float noiseMuLimit = 1; float noiseSigmaLimit = 1; float brightContrastLimit = 0.2; float brightnessLimit = 0;
//	std::map<std::string, float> name_idx = {};
//public:
//	DetDataset(std::vector<std::string> images, std::vector<std::string> labels, std::vector<std::string> class_names, bool istrain = true, int width_ = 416, int height_ = 416, float hflip_prob = 0.5, float vflip_prob = 0)
//	{
//		list_images = images; list_labels = labels; isTrain = istrain; width = width_; height = height_;
//		hflipProb = hflip_prob; vflipProb = vflip_prob;
//		for (int i = 0; i < class_names.size(); ++i)
//		{
//			name_idx.insert(std::pair<std::string, float>(class_names[i], float(i)));
//		}
//	}
//
//	torch::data::Example<> get(size_t index) override {
//		std::string image_path = list_images.at(index);
//		std::string annotation_path = list_labels.at(index);
//
//		cv::Mat img = cv::imread(image_path, 1);
//		std::vector<BBox> boxes = loadXML(annotation_path);
//		Data m_data(img, boxes);
//
//		m_data = Augmentations::Resize(m_data, width, height, 1);
//
//		float width_under1 = 1.0 / m_data.image.cols;
//		float height_under1 = 1.0 / m_data.image.rows;
//		torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 });
//
//		int box_num = m_data.bboxes.size();
//		if (m_data.bboxes.size() == 0)
//		{
//			torch::Tensor label_tensor = torch::ones({ 1 });
//			return { img_tensor.clone(), label_tensor.clone() };
//		}
//		torch::Tensor label_tensor = torch::zeros({ box_num, 5 }).to(torch::kFloat32);
//		for (int i = 0; i < box_num; ++i)
//		{
//			label_tensor[i][2] = m_data.bboxes[i].GetW() * width_under1;
//			label_tensor[i][3] = m_data.bboxes[i].GetH() * height_under1;
//			label_tensor[i][0] = m_data.bboxes[i].xmin * width_under1 + label_tensor[i][2] / 2;
//			label_tensor[i][1] = m_data.bboxes[i].ymin * height_under1 + label_tensor[i][3] / 2;
//			label_tensor[i][4] = name_idx.at(m_data.bboxes[i].name);
//		}
//		return { img_tensor.clone(), label_tensor.clone() };
//	}
//
//	std::vector<ExampleType> get_batch(c10::ArrayRef<size_t> indices) override {
//		std::vector<ExampleType> batch;
//		batch.reserve(indices.size());
//		for (const auto i : indices) {
//			batch.push_back(get(i));
//		}
//		return batch;
//	}
//
//	torch::optional<size_t> size() const override {
//		return list_labels.size();
//	}
//};
//
//inline bool does_exist(const std::string& name) {
//	struct stat buffer;
//	return (stat(name.c_str(), &buffer) == 0);
//}

// Use Dataset 
//int width = 416;
//int height = 416;
//std::vector<std::string> name_list;
//
//
//std::string data_path = "C:/Users/admin/Desktop/Projects/TorchCPP_EfficientDet/Resources/dataset";
//
//std::string name_list_path = data_path + "/voc_classes.txt";
//
//std::ifstream ifs;
//ifs.open(name_list_path, std::ios::in);
//if (!ifs.is_open())
//{
//	std::cout << "Open " << name_list_path << " file failed." << std::endl;
//	return -1;
//}
//std::string buf = "";
//while (getline(ifs, buf))
//{
//	name_list.push_back(buf);
//}
//
//int num_classes = name_list.size();
//
//std::string train_labels_path = data_path + "/train/labels";
//std::string train_images_path = data_path + "/train/images";
//std::string val_labels_path = data_path + "/val/labels";
//std::string val_images_path = data_path + "/val/images";
//
//std::vector<std::string> list_images_train = {};
//std::vector<std::string> list_labels_train = {};
//std::vector<std::string> list_images_val = {};
//std::vector<std::string> list_labels_val = {};
//
//std::string image_type = ".jpg";
//int num_epochs = 30;
//int batch_size = 4;
//float learning_rate = 0.001;
//
//load_det_data_from_folder(train_labels_path, train_images_path, image_type, list_images_train, list_labels_train);
//load_det_data_from_folder(val_labels_path, val_images_path, image_type, list_images_val, list_labels_val);
//
//if (list_images_train.size() < batch_size || list_images_val.size() < batch_size) {
//	std::cout << "Image numbers less than batch size or empty image folder" << std::endl;
//	return -1;
//}
//
//if (!does_exist(list_images_train[0]))
//{
//	std::cout << "Image path is invalid get first train image" << list_images_train[0] << std::endl;
//	return -1;
//}
//auto custom_dataset_train = DetDataset(list_images_train, list_labels_train, name_list, true, width, height);
//auto custom_dataset_val = DetDataset(list_images_val, list_labels_val, name_list, false, width, height);
//auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), batch_size);
//auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);

// Model Train
//return 0;
//YOLOv4 model = YOLOv4(CSPDarknet53(num_classes, true));
//model->to(device);
//
//auto pretrained_dict = model->named_parameters();
//auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();
//for (int epoc_count = 0; epoc_count < num_epochs; ++epoc_count)
//{
//	float loss_sum = 0.0f;
//	int batch_count = 0;
//	float loss_train = 0.0f;
//	float loss_val = 0.0f;
//	float best_loss = 1e10;
//
//	if (epoc_count == int(num_epochs / 2)) { learning_rate /= 10; }
//	torch::optim::Adam optimizer(model->parameters(), learning_rate);
//	//if (epoc_count < int(num_epochs / 10))
//	//{
//	//	for (auto mm : pretrained_dict)
//	//	{
//	//		// if (strstr(mm.key().c_str(), "")
//	//	}
//	//}
//	model->train();
//	for (auto& batch : *data_loader_train) {
//		std::vector<torch::Tensor> images_vec = {};
//		std::vector<torch::Tensor> targets_vec = {};
//		if (batch.size() < batch_size) continue;
//		for (int i = 0; i < batch_size; ++i)
//		{
//			images_vec.push_back(batch[i].data.to(FloatType));
//			targets_vec.push_back(batch[i].target.to(FloatType));
//		}
//		auto data = torch::stack(images_vec).div(255.0);
//
//		optimizer.zero_grad();
//		auto outputs = model->forward(data);
//
//		std::cout << "large : " << outputs[0].sizes() << std::endl;
//		std::cout << "medium : " << outputs[1].sizes() << std::endl;
//		std::cout << "small : " << outputs[2].sizes() << std::endl;
//	}
//}