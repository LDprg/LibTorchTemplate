#pragma once
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <ctime>

class TBFGenerator
{
public:
	TBFGenerator(std::string logdir, std::string name = "run1", bool usedate = true)
	{
		std::string path = "/tfevents.";

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
};