#include <torch\torch.h>

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
		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
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

	torch::Tensor tensor = torch::randn({ 100,1 }, device_type);
	torch::Tensor tensorbak = torch::randn({ 100,1 }, device_type);
	torch::optim::SGD optimizer(net->parameters(), 0.01);

	for (int i = 0; i < 20000; ++i)
	{
		optimizer.zero_grad();
		tensor = torch::randn({ 100,1 }, device_type)*M_PI*2;
		tensorbak = torch::sin(tensor);

		torch::Tensor prediction = net->forward(tensor);
		torch::Tensor loss = torch::mse_loss(prediction, tensorbak);

		loss.backward();

		optimizer.step();

		if(i%1000==0)
			std::cout << "Generation: " << i << "\t" << loss.item<double>() << std::endl;
	}

	for (double j = 0; j < M_PI * 2; j += 0.1)
	{
		tensor[0][0] = j;
		torch::Tensor prediction = net->forward(tensor);

		std::cout << j << "\t" << prediction[0][0].item<double>() << "\t" << sin(j) << std::endl;
		std::ofstream ofs("./graph.txt", std::ofstream::app);

		ofs << j << ';' << prediction[0][0].item<double>() << ';' << sin(j) << std::endl;

		ofs.close();
	}
}