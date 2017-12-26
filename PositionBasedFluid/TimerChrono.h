#pragma once
#ifndef __TIMERCHRONO_H__
#define __TIMERCHRONO_H__

#include <iostream>
#include <chrono>

using namespace std;

class TimerChrono
{
public:
	TimerChrono()
	{
		acc_duration.zero();
		acc_count = 0;
	}
	void start()
	{
		start1 = std::chrono::system_clock::now();
	}

	void end(char* p_name)
	{
		end1 = std::chrono::system_clock::now();
		duration = (end1 - start1);
		printf("%s Cost : %lf sec \n", p_name, duration);
		acc_duration += duration;
		acc_count += 1;
	}

	void printAvg(char* p_name)
	{
		printf("%s Avg Cost : %lf sec \n", p_name, (acc_duration / (double)acc_count));
	}

private:
	chrono::system_clock::time_point start1, end1;
	chrono::duration<double> duration;
	chrono::duration<double> acc_duration;
	int acc_count;

};

#endif