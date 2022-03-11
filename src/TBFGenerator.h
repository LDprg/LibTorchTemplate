#pragma once
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <ctime>

class TBFGenerator
{
public:
	TBFGenerator(std::string logdir, bool autoinc = true, std::string name = "run", bool usedate = true)
	{
		std::string path = "/tfevents.";

		if (autoinc)
			name += std::to_string(getInc(logdir, name));

		std::filesystem::create_directories(logdir);
		std::filesystem::create_directories(logdir + "/" + name);
		path = "/" + name + path;

		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		std::ostringstream oss;

		if(usedate)
			oss << std::put_time(&tm, "%d-%m-%Y.%H-%M-%S.");
		
		auto date = oss.str();

		data = "./" + logdir + path +  date  + name;
	}

	operator std::string() const { return data; }
	operator const char* () const { return data.c_str(); }

	const char * str() const { return data.c_str(); }

private:
	std::string data = "";

	int getInc(std::string logdir, std::string name)
	{
		int num = 0;
		for (const auto& entry : std::filesystem::directory_iterator(logdir))
		{
			std::string fname = entry.path().string().substr(entry.path().string().find_last_of("/\\") + 1);
			if (fname.find(name) != std::string::npos)
			{
				//std::cout << entry.path() << "|" << fname << "|" << fname.substr(fname.find_last_of(name) + 1) << std::endl;
				if (!fname.substr(fname.find_last_of(name) + 1).empty())
				{
					int fnum = std::stoi(fname.substr(fname.find_last_of(name) + 1));
					if (num < fnum)
						num = fnum;
				}
			}
		}
		++num;
		return num;
	}
};