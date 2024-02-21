#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
    std::vector<std::tuple<std::string, int64_t>> annotations;

public:
    explicit CustomDataset()
    {
        for (const fs::directory_entry& entry :
            fs::recursive_directory_iterator(
                "C:/Users/admin/Desktop/Projects/TorchCPP_EfficientDet/Resources/data/apples")) {
            annotations.push_back(std::make_tuple(entry.path().string(), 0));
        }

        for (const fs::directory_entry& entry :
            fs::recursive_directory_iterator(
                "C:/Users/admin/Desktop/Projects/TorchCPP_EfficientDet/Resources/data/bananas")) {
            annotations.push_back(std::make_tuple(entry.path().string(), 1));
        }
    }

    torch::data::Example<> get(size_t index) override {
        std::string file_location = std::get<0>(annotations[index]);
        int64_t label = std::get<1>(annotations[index]);

        cv::Mat img = cv::imread(file_location);

        torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).clone();
        img_tensor = img_tensor.permute({ 2, 0, 1 });

        torch::Tensor label_tensor = torch::full({ 1 }, label);

        return { img_tensor, label_tensor };
    }

    torch::optional<size_t> size() const override {
        return annotations.size();
    }
};

struct ConvNetImpl : public torch::nn::Module
{
    ConvNetImpl(int64_t channels, int64_t height, int64_t width)
        :
        conv1(torch::nn::Conv2dOptions(3, 8, 5).stride(2)),
        conv2(torch::nn::Conv2dOptions(8, 16, 3).stride(2)),
        n(GetConvOutput(channels, height, width)),
        lin1(n, 32),
        lin2(32, 2)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("lin1", lin1);
        register_module("lin2", lin2);
    }

    torch::Tensor forward(torch::Tensor x) {

        x = torch::relu(torch::max_pool2d(conv1(x), 2));
        x = torch::relu(torch::max_pool2d(conv2(x), 2));

        x = x.view({ -1, n });

        x = torch::relu(lin1(x));
        x = torch::log_softmax(lin2(x), 1);

        return x;
    }

    int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {
        torch::Tensor x = torch::zeros({ 1, channels, height, width });
        x = torch::max_pool2d(conv1(x), 2);
        x = torch::max_pool2d(conv2(x), 2);

        return x.numel();
    }

    torch::nn::Conv2d conv1, conv2;
    int64_t n;
    torch::nn::Linear lin1, lin2;
};

TORCH_MODULE(ConvNet);


int main_tutorial()
{
    ConvNet model(3, 64, 64);

    auto data_set = CustomDataset().map(torch::data::transforms::Stack<>());

    int64_t batch_size = 1;
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(data_set, batch_size);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    int64_t n_epochs = 10;
    int64_t log_interval = 10;
    int dataset_size = data_set.size().value();

    float best_mse = std::numeric_limits<float>::max();

    for (int epoch = 1; epoch <= n_epochs; ++epoch)
    {
        size_t batch_idx = 0;
        float mse = 0.0f;
        int count = 0;

        for (auto& batch : *data_loader) {
            auto imgs = batch.data;
            auto labels = batch.target.squeeze();

            imgs = imgs.to(torch::kF32);
            labels = labels.to(torch::kInt64);

            optimizer.zero_grad();
            auto output = model(imgs);
            auto loss = torch::nll_loss(output, labels);

            loss.backward();
            optimizer.step();

            mse += loss.template item<float>();

            batch_idx++;
            if (batch_idx % log_interval == 0)
            {
                std::printf(
                    "\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f",
                    epoch,
                    n_epochs,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>());
            }

            count++;
        }

        mse /= (float)count;
        printf(" Mean squared error: %f\n", mse);

        if (mse < best_mse)
        {
            torch::save(model, "C:/Users/admin/Desktop/Projects/TorchCPP_EfficientDet/Resources/models/best_model.pt");
            best_mse = mse;
        }
    }

    return 0;
}