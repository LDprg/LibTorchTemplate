#include <torch\torch.h>

struct Net : public torch::nn::Module
{
	torch::nn::Linear fc1{nullptr}, fc2{ nullptr }, fc3{ nullptr };

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
	auto net = std::make_shared<Net>();

	net->to(torch::kCPU);

	torch::Tensor tensor = torch::randn({ 1,1 });
	torch::Tensor tensorbak = torch::randn({ 1,1 });
	torch::Tensor temp;
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.001));

	for (int i = 0; i < 100; ++i)
	{
		for (double j = 0; j < M_PI*2; j+=0.1)
		{
			optimizer.zero_grad();
			tensor[0][0] = j;
			tensorbak[0][0] = sin(j);

			torch::Tensor prediction = net->forward(tensor);
			torch::Tensor loss = torch::mse_loss(prediction, tensorbak);
			temp = loss;

			net->to(torch::kCUDA);
			loss.backward();
			net->to(torch::kCPU);

			optimizer.step();
			if (i == 99)
			{
				std::cout << j << "\t" << prediction[0][0].item<double>() << "\t" << sin(j) << std::endl;
				std::ofstream ofs("./graph.txt", std::ofstream::app);

				ofs << j << ';' << prediction[0][0].item<double>() << ';' << sin(j) << std::endl;

				ofs.close();
			}
		}
		std::cout << "Generation: " << i << "\t" << temp.item<float>() << std::endl;
	}
}