#pragma once
#ifndef __FLOWBOUNDARY_H__
#define __FLOWBOUNDARY_H__

#include <Eigen/dense>
#include <vector>
#include "FluidWorld.h"

using namespace Eigen;

class GridCell
{
public:
	GridCell()
	{
		m_centerPosition[0] = m_centerPosition[1] = m_centerPosition[2] = 0.0f;
		m_gridSize = 0.0f;
	}
	void SetValue(Vector3f p_cp, float p_gridSize)
	{
		m_centerPosition[0] = p_cp[0];
		m_centerPosition[1] = p_cp[1];
		m_centerPosition[2] = p_cp[2];

		m_gridSize = p_gridSize;

		float halfGridSize = 0.5f * m_gridSize;
		m_minCorner[0] = m_centerPosition[0] - halfGridSize;
		m_minCorner[1] = m_centerPosition[1] - halfGridSize;
		m_minCorner[2] = m_centerPosition[2] - halfGridSize;

		m_maxCorner[0] = m_centerPosition[0] + halfGridSize;
		m_maxCorner[1] = m_centerPosition[1] + halfGridSize;
		m_maxCorner[2] = m_centerPosition[2] + halfGridSize;
	}
	
	Vector3f m_centerPosition;
	Vector3f m_minCorner;
	Vector3f m_maxCorner;
	Vector3f m_avgVel;
	
	float m_gridSize;
	bool m_inFluid;

	std::vector<Vector3f> m_boundaryNeighborList;

};

class FlowBoundary
{
public:
	FlowBoundary()
	{
		m_gridSize = 0.0f;
		m_boundarySize = 0;
		m_fThr = 1.0f;
		m_lcp = 0.0f;

	}
	~FlowBoundary()
	{
		m_boundary.clear();
		m_boundary.resize(0);
	}

	void CreateFlowBoundary(Vector3f start, Vector3f end, float gridSize);
	void SearchBoundaryNeighbor(std::vector<Vector3f>& p_boundaryParticles, float p_radius);
	void SetFluidThreshold(float p_threshold);
	
	std::vector<GridCell>& GetFlowBoundary()	{	return m_boundary;	}
	int GetBoundarySize() { return m_boundarySize; }
	float GetGridSize() { return m_gridSize; }

	void CreateFinePs(std::vector<FParticle*>& p_coarseP, float p_coarseP_radius, std::vector<FParticle*>& p_fineP, float p_fineP_radius);
	
	std::vector<Vector3f>& GetCPSet() { return m_CPSet; }
	std::vector<Vector3f>& GetAvgVelSet() { return m_avgVelSet; }

private:
	void CreateBoundaryWall(Vector3f p_min, Vector3f p_max);

	float m_gridSize;
	float m_lcp;
	int m_cntOfcandisPerOneAxis;

	std::vector<GridCell> m_boundary;
	std::vector<Vector3f> m_CPSet;
	std::vector<Vector3f> m_avgVelSet;
	int m_boundarySize;
	float m_fThr;
};


#endif