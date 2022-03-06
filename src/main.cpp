#include <torch\torch.h>

struct Net : public torch::nn::Module
{
	torch::nn::Linear fc1{nullptr}, fc2{ nullptr }, fc3{ nullptr };

	Net()
	{
		fc1 = register_module("fc1", torch::nn::Linear(1, 512));
		fc2 = register_module("fc2", torch::nn::Linear(512, 512));
		fc3 = register_module("fc3", torch::nn::Linear(512, 1));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::tanh(fc1->forward(x));
		x = torch::tanh(fc2->forward(x));
		x = torch::dropout(x, 0.5, true);
		x = fc3->forward(x);

		return x;
	}
};

auto main() -> int
{
	int epoch = 30000;

	std::remove("./graph.txt");

	torch::Device device_type = torch::kCUDA;

	auto net = std::make_shared<Net>();

	net->to(device_type);
		
	torch::Tensor tensor = torch::rand({ 1, 1 }, device_type) * M_PI * 2; //torch::range(0, M_PI * 2, 0.1,device_type);
	torch::Tensor tensortar = torch::sin(tensor);

	torch::optim::SGD optimizer(net->parameters(), 0.001);

	for (int i = 0; i < epoch; ++i)
	{		
		//tensor = torch::rand({ 1,1 }, device_type) * M_PI * 2;
		tensor[0][0] = (M_PI * 2)*((double)i/(double)epoch);
		tensortar = torch::sin(tensor);

		optimizer.zero_grad();

		torch::Tensor prediction = net->forward(tensor);
		torch::Tensor loss = torch::mse_loss(prediction, tensortar);

		loss.backward();

		optimizer.step();

		if (i == 0)
		{
			for (double j = 0; j < M_PI * 2; j += 0.1)
			{
				std::cout << j << "\t" << prediction[0][0].item<double>() << "\t" << sin(j) << std::endl;
			}
		}

		if (i % (epoch/10) == 0)
		{
			std::cout << "Generation: " << i << "\t" << loss.item<double>() << std::endl;
			std::cout << tensortar[0][0].item<double>() << std::endl << prediction[0][0].item<double>() << std::endl << std::endl;
			//std::cout << tensortar[1][0].item<double>() << std::endl << prediction[1][0].item<double>() << std::endl << std::endl;
		}
		
		if (i == epoch-1)
		{
			std::ofstream ofs("./graph.txt", std::ofstream::app);
			for (double j = 0; j < M_PI * 2; j += 0.1)
			{
				std::cout << j << "\t" << prediction[0][0].item<double>() << "\t" << sin(j) << std::endl;

				ofs << j << ';' << prediction[0][0].item<double>() << ';' << sin(j) << std::endl;
			}
			ofs.close();
		}
	}
}