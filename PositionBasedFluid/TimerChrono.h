#pragma once
#ifndef __TIMERCHRONO_H__
#define __TIMERCHRONO_H__

#include <iostream>
#include <chrono>

using namespace std;

class TimerChrono
{
public:
	void start()
	{
		start1 = std::chrono::system_clock::now();
	}

	void end(char* p_name)
	{
		end1 = std::chrono::system_clock::now();
		duration = (end1 - start1);
		printf("%s Cost : %lf sec \n", p_name, duration);
	}

private:
	chrono::system_clock::time_point start1, end1;
	chrono::duration<double> duration;
};

#endif