#include <sstream>

#include <torch\torch.h>
#include <tensorboard_logger.h>

#include "TBFGenerator.h"

struct Net : public torch::nn::Module
{
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };

	Net()
	{
		fc1 = register_module("fc1", torch::nn::Linear(1, 1024));
		fc2 = register_module("fc2", torch::nn::Linear(1024, 1024));
		fc3 = register_module("fc3", torch::nn::Linear(1024, 1));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::leaky_relu(fc1->forward(x));
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
	
	TensorBoardLogger TBL(TBFGenerator("runs", "run5"));

	torch::Tensor tensor;
	torch::Tensor tensortar;
	torch::optim::SGD optimizer(net->parameters(), 0.001);

	for (int i = 0; i < 20000; ++i)
	{
		optimizer.zero_grad();
		tensor = torch::rand({200,1}, device_type) * M_PI * 2;
		tensortar = torch::sin(tensor);

		torch::Tensor prediction = net->forward(tensor);
		torch::Tensor loss = torch::mse_loss(prediction, tensortar);

		loss.backward();

		optimizer.step();

		TBL.add_scalar("Loss", i, loss.item<double>());

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
		TBLS.add_scalar("Output", i, sin(j));

		std::cout << j << "\t" << prediction[0][0].item<double>() << "\t" << sin(j) << std::endl;
		std::ofstream ofs("./graph.txt", std::ofstream::app);

		ofs << j << ';' << prediction[0][0].item<double>() << ';' << sin(j) << std::endl;

		ofs.close();
		++i;
	}
}