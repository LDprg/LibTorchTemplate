#include <sstream>
#include <regex>

#include <torch\torch.h>
#include <tensorboard_logger.h>

#include "TBFGenerator.h"

template <class T>
std::vector<T> toVector(torch::Tensor& tin)
{
	torch::Tensor t = tin.cpu();
	return std::vector<T>(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
}

struct Net : public torch::nn::Module
{
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };

	Net()
	{
		fc1 = register_module("fc1", torch::nn::Linear(1, 1024));
		fc2 = register_module("fc2", torch::nn::Linear(1024, 1024));
		fc3 = register_module("fc3", torch::nn::Linear(1024, 1));

		if (auto* linear = as<torch::nn::Linear>())
		{
			torch::nn::init::xavier_normal_(linear->weight);
			torch::nn::init::constant_(linear->bias, 0.01);
		}
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::tanh(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);

		return x;
	}
};

auto main() -> int
{
	torch::Device device_type = torch::kCUDA;

	std::remove("./graph.txt");

	auto net = std::make_shared<Net>();
	net->to(device_type);
	
	TensorBoardLogger TBL(TBFGenerator("runs", "run9"));

	torch::Tensor tensor;
	torch::Tensor tensortar;
	torch::optim::SGD optimizer(net->parameters(), 0.001);

	for (int i = 0; i < 20000; ++i)
	{
		optimizer.zero_grad();
		tensor = torch::rand({20,1}, device_type) * M_PI * 2;
		tensortar = torch::sin(tensor);

		torch::Tensor prediction = net->forward(tensor);
		torch::Tensor loss = torch::mse_loss(prediction, tensortar);

		if (i%500==0)
		{
			for (std::string j : net->named_parameters().keys())
				TBL.add_histogram(std::regex_replace(j, std::regex("[.]"), "/"), i, toVector<double>(net->named_parameters()[j]));
		}

		loss.backward();

		optimizer.step();

		TBL.add_scalar("Running/Loss", i, loss.item<double>());
		TBL.add_scalar("Running/Accuracy", i, torch::sum(tensortar == prediction).item<double>());

		if (i % 1000 == 0)
		{
			std::cout << "Generation: " << i << "\t" << loss.item<double>() << std::endl;
		}
	}

	TensorBoardLogger TBLS(TBFGenerator("runs", "sine", false));
	int i = 0;
	for (double j = 0; j < M_PI * 2; j += 0.1)
	{
		tensor[0][0] = j;
		torch::Tensor prediction = net->forward(tensor);

		TBL.add_scalar("Output", i, prediction[0][0].item<double>());
		TBL.add_scalar("Output/Accuracy", i, abs(sin(j)-prediction[0][0].item<double>()));
		TBLS.add_scalar("Output", i, sin(j));

		std::cout << j << "\t" << prediction[0][0].item<double>() << "\t" << sin(j) << std::endl;
		std::ofstream ofs("./graph.txt", std::ofstream::app);

		ofs << j << ';' << prediction[0][0].item<double>() << ';' << sin(j) << std::endl;

		ofs.close();
		++i;
	}
}