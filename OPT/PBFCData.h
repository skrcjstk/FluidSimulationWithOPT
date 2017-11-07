#pragma once
#ifndef __PBFCONTROL_H__
#define __PBFCONTROL_H__

#include <Eigen/dense>
#include <vector>

using namespace Eigen;

class PBFControlData
{
public:
	std::vector<float> m_lambdaForCoarse;
	std::vector<Vector3f> m_corrWithDensity;
	
	std::vector<float> m_weightForVelocityC;
	std::vector<Vector3f> m_corrWithVelocity;
};

#endif