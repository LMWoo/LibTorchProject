#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

// #define TENSOR_TEST
// define FASTER_RCNN
#define YOLO

std::vector<torch::Tensor> non_max_suppression(torch::Device device, torch::Tensor preds, float score_thresh = 0.5, float iou_thresh = 0.5)
{
    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;


        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor  dets = pred.slice(1, 0, 6);
        torch::Tensor keep = torch::empty({ dets.sizes()[0] }).to(device);
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));

        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);

        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());


            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1).to(device);


            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }

            torch::Tensor overlaps = widths * heights;

            // FIlter by IOUs

            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
        std::cout << "=============================" << std::endl;
        std::cout << torch::index_select(dets, 0, keep.slice(0, 0, count)) << std::endl;
        std::cout << "=============================" << std::endl;
    }
    return output;
}

int main()
{
#ifdef TENSOR_TEST
    torch::Tensor tensor = torch::randn({ 4, 4 });
    std::cout << tensor << std::endl;
    std::cout << tensor.select(0, 3) << std::endl;

    return 0;
#endif

#ifdef FASTER_RCNN
    //torch::Device device(torch::kCPU);
    //torch::jit::script::Module model;

    //if (torch::cuda::is_available())
    //{
    //    device = torch::Device(torch::kCUDA);
    //}

    //try {
    //    model = torch::jit::load("C:/Users/admin/Desktop/Projects/TorchCPP_YOLO/Resources/yolov5s.torchscript.pt");
    //}
    //catch (const c10::Error& e) {
    //    std::cerr << "error loading the model\n";
    //    return -1;
    //}

    //std::vector<torch::jit::IValue> inputs;
    //std::vector<torch::Tensor> images;

    //images.push_back(torch::rand({ 3, 256, 275 } ));
    //images.push_back(torch::rand({ 3, 256, 275 }));

    //inputs.push_back(images);

    //auto output = model.forward(inputs);
    //std::cout << output << std::endl;

    torch::DeviceType device_type;
    device_type = torch::kCPU;

    torch::jit::script::Module module;
    try {
        std::cout << "Loading model\n";
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("C:/Users/admin/Desktop/Projects/TorchCPP_YOLO/Resources/fasterrcnn_resnet50_fpn.pt");
        std::cout << "Model loaded\n";
    }
    catch (const torch::Error& e) {
        std::cout << "error loading the model\n";
        return -1;
    }
    catch (const std::exception& e) {
        std::cout << "Other error: " << e.what() << "\n";
        return -1;
    }

    // TorchScript models require a List[IValue] as input
    std::vector<torch::jit::IValue> inputs;

    // Faster RCNN accepts a List[Tensor] as main input
    std::vector<torch::Tensor> images;
    images.push_back(torch::rand({ 3, 256, 275 }));
    images.push_back(torch::rand({ 3, 256, 275 }));

    inputs.push_back(images);
    auto output = module.forward(inputs);

    std::cout << "ok\n";
    std::cout << "output" << output << "\n";

    if (torch::cuda::is_available()) {
        // Move traced model to GPU
        module.to(torch::kCUDA);

        // Add GPU inputs
        images.clear();
        inputs.clear();

        torch::TensorOptions options = torch::TensorOptions{ torch::kCUDA };
        images.push_back(torch::rand({ 3, 256, 275 }, options));
        images.push_back(torch::rand({ 3, 256, 275 }, options));

        inputs.push_back(images);
        auto output = module.forward(inputs);

        std::cout << "ok\n";
        std::cout << "output" << output << "\n";
    }




#endif

#ifdef YOLO
	torch::Device device(torch::kCPU);
	torch::jit::script::Module model;
	
	try {
		model = torch::jit::load("C:/Users/admin/Desktop/Projects/TorchCPP_YOLO/Resources/yolov5s.torchscript.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	model.to(device);
	model.eval();

	std::vector<std::string> classnames;
	std::ifstream f("C:/Users/admin/Desktop/Projects/TorchCPP_YOLO/Resources/coco.names");
	std::string name = "";
	while (std::getline(f, name))
	{
		classnames.push_back(name);
	}

	std::string imagePath = "C:/Users/admin/Desktop/Projects/TorchCPP_YOLO/Resources/humans.jpg";
	cv::Mat matImage = cv::imread(imagePath);
	cv::Mat resizedImage;

	cv::resize(matImage, resizedImage, cv::Size(640, 384));
	cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
	torch::Tensor imgTensor = torch::from_blob(resizedImage.data, { resizedImage.rows, resizedImage.cols, 3 }, torch::kByte);
	imgTensor = imgTensor.permute({ 2, 0, 1 });
	imgTensor = imgTensor.toType(torch::kFloat);
	imgTensor = imgTensor.div(255.0);
	imgTensor = imgTensor.unsqueeze(0);
	imgTensor = imgTensor.to(device);

	std::vector<torch::jit::IValue> vecInputs;
	vecInputs.push_back(imgTensor);

	torch::Tensor preds = model.forward({ imgTensor }).toTuple()->elements()[0].toTensor();


    /*if (torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
    }*/
    preds = preds.to(device);

    // std::vector<torch::Tensor> dets = non_max_suppression(device, preds, 0.2, 0.5);
    std::cout << "start nms" << std::endl;
    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > 0.2).select(1, 0));
        if (pred.sizes()[0] == 0) continue;


        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor  dets = pred.slice(1, 0, 6);
        torch::Tensor keep = torch::empty({ dets.sizes()[0] }).to(device);
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));


        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);

        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());


            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1).to(device);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1).to(device);


            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }

            torch::Tensor overlaps = widths * heights;

            // FIlter by IOUs

            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);

            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= 0.5).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
        std::cout << "=============================" << std::endl;
        std::cout << torch::index_select(dets, 0, keep.slice(0, 0, count)) << std::endl;
        std::cout << "=============================" << std::endl;
    }

    if (output.size() > 0)
    {
        for (int i = 0; i < output[0].sizes()[0]; ++i)
        {
            float left = output[0][i][0].item().toFloat() * matImage.cols / 640;
            float top = output[0][i][1].item().toFloat() * matImage.rows / 384;
            float right = output[0][i][2].item().toFloat() * matImage.cols / 640;
            float bottom = output[0][i][3].item().toFloat() * matImage.rows / 384;
            float score = output[0][i][4].item().toFloat();
            int classID = output[0][i][5].item().toInt();

            cv::rectangle(matImage, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);
            cv::putText(matImage, classnames[classID] + ": " + cv::format("%.2f", score),
                cv::Point(left, top),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
    
    
    cv::imshow("matImage", matImage);
    cv::waitKey();
    cv::destroyAllWindows();
#endif
	return 0;
}