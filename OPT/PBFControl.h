#pragma once
#ifndef __PBFCONTROL_H__
#define __PBFCONTROL_H__

#include <Eigen/dense>
#include <vector>
#include "FluidWorld.h"
using namespace Eigen;

class TrainData
{
public:
	float weight;
	Vector3f RVec;
	Vector3f RVel;
};
class PBFControlData
{
public:
	std::vector<float> m_lambdaForCoarse;
	std::vector<Vector3f> m_corrWithDensity;
	std::vector<float> m_weightForVelocityC;
	std::vector<Vector3f> m_corrWithVelocity;
};
class PBFControl
{
public:
	FluidKernel k;

	// for TrainingData
	std::vector<std::vector<TrainData>> m_tDataForMain;
	//std::vector<std::vector<TrainData>> m_tDataForMainBoundary;
	std::vector<std::vector<TrainData>> m_tDataForSub;
	std::vector<Vector3f> m_deltaPWithControl;

	
	// for PBFC
	PBFControlData m_PBFCData;

	std::vector<std::vector<int>> m_neighListwithSubP;
	std::vector<std::vector<int>> m_neighListwithSubBoundaryP;

	void Initialize(FluidWorld* p_mainWorld, FluidWorld* p_subWorld);
	void NeighborBTWTwoResForPBFC(FluidWorld* p_mainWorld, FluidWorld* p_subWorld);
	void SolvePBFCConstaints(FluidWorld* p_mainWorld, FluidWorld* p_subWorld);
	void UpdateTrainingDataForMain(FluidWorld* p_mainWorld, FluidWorld* p_subWorld);
	void UpdateTrainingDataForSub(FluidWorld* p_subWorld);
	
	float m_intensityOfDensityC = 1.0f;
	float m_intensityOfVelocityC = 0.0f;

};

#endif
