#include "FlowBoundary.h"

void FlowBoundary::CreateFlowBoundary(Vector3f start, Vector3f end, float p_fineR)
{
	m_gridSize = 2.0f * p_fineR;
	m_lcp = m_gridSize * 0.01f;
	m_cntOfcandisPerOneAxis = int((m_gridSize - m_lcp) / m_lcp) + 1;

	CreateBoundaryWall(start, end);
	m_boundarySize = m_boundary.size();

	//float coarseR = 2.0f * p_fineR;
	k.SetSmoothingRadius(4.0f * p_fineR);
	
}

void FlowBoundary::SetFluidThreshold(float p_thr)
{
	m_fThr = p_thr;
}

void FlowBoundary::CreateBoundaryWall(Vector3f p_min, Vector3f p_max)
{
	Vector3f diff = p_max - p_min;

	unsigned int countX = (unsigned int)(diff[0] / m_gridSize) + 1;
	unsigned int countY = (unsigned int)(diff[1] / m_gridSize) + 1;
	unsigned int countZ = (unsigned int)(diff[2] / m_gridSize) + 1;

	printf("Boundary Wall - diff(%f, %f, %f)\n", diff[0], diff[1], diff[2]);
	printf("Boundary Wall - count(%d, %d, %d)\n", countX, countY, countZ);

	unsigned int startIndex = m_boundary.size();
	m_boundary.resize(startIndex + countX*countY*countZ);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (unsigned int i = 0; i < countX; i++)
		{
			for (unsigned int j = 0; j < countY; j++)
			{
				for (unsigned int k = 0; k < countZ; k++)
				{
					Vector3f position = p_min + Vector3f(i*m_gridSize, j*m_gridSize, k*m_gridSize);
					m_boundary[startIndex + i*countY*countZ + j*countZ + k].SetValue(position, m_gridSize);
				}
			}
		}
	}
}

void FlowBoundary::SearchBoundaryNeighbor(std::vector<Vector3f>& p_boundaryParticles, float p_fineR)
{
	float supportRadius = k.GetSmoothingRadius();

	for (int i = 0; i < m_boundarySize; i++)
	{
		m_boundary[i].m_boundaryNeighborList.resize(0);

		for (int j = 0; j < p_boundaryParticles.size(); j++)
		{
			float dist = (p_boundaryParticles[j] - m_boundary[i].m_centerPosition).norm();
			if (dist < supportRadius)
			{
				m_boundary[i].m_boundaryNeighborList.push_back(Vector3f(p_boundaryParticles[j][0], p_boundaryParticles[j][1], p_boundaryParticles[j][2]));
			}
		}
	}
}

void FlowBoundary::CreateFinePs(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	std::vector<Vector3f> neighborCoarseParticles;
	float R = p_mainWorld->GetParticleRadius();
	float r = p_subWorld->GetParticleRadius();
	float searchRange = 4.0f * R;
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	float Vcell = m_gridSize * m_gridSize * m_gridSize;

	for (int i = 0; i < m_boundarySize; i++)
		m_boundary[i].m_inFluid = false;
		
	for (int i = 0; i < m_boundarySize; i++) {
		float m_fluidRatio = 0.0f;
		for (int j = 0; j < coarseP.size(); j++) {
			float dist = (coarseP[j]->m_curPosition - m_boundary[i].m_centerPosition).norm();
			if (dist <= searchRange) {
				// overlap ratio
				bool isOverlap = true;
				Vector3f length;
				Vector3f Vl = m_boundary[i].m_minCorner - coarseP[j]->m_curPosition;
				Vector3f Vr = m_boundary[i].m_maxCorner - coarseP[j]->m_curPosition;

				for (int k = 0; k < 3; k++) {
					float dmax = std::max(fabsf(Vl[k]), fabsf(Vr[k]));
					float dmin = std::min(fabsf(Vl[k]), fabsf(Vr[k]));

					if (dmin >= R) {
						isOverlap = false;
						break;
					}
					else {
						if (Vl[k] * Vr[k] >= 0) {
							if (dmax > R)
								length[k] = R - dmin;
							else
								length[k] = 2.0f * r;
						}
						else {
							if (dmax > R)
								length[k] = R + dmin;
							else
								length[k] = 2.0f * r;
						}
					}
				}
				float fluidRatio = length[0] * length[1] * length[2] / Vcell;

				if (isOverlap == true && m_fluidRatio >= 0.0f) {
					m_fluidRatio += fluidRatio;
					break;
				}
			}
		}
		if(m_fluidRatio >= m_fThr)
			m_boundary[i].m_inFluid = true;
	}
	
	for (int i = 0; i < m_boundarySize; i++)
		if (m_boundary[i].m_inFluid == true)
			p_subWorld->AddFParticle(m_boundary[i].m_centerPosition, Vector3f(0.0f, 0.0f, 0.0f));

	m_neighborListBTW.resize(p_subWorld->GetNumOfParticles());
	m_tempVelforCoarse.resize(p_mainWorld->GetNumOfParticles());

	/*
	// search neighbor fineP
	float searchArea = 3.0f * p_fineP_radius;
	std::vector<int> neighborIdx;

	for (int j = 0; j < p_fineP.size(); j++)
	{
	if ((m_boundary[i].m_centerPosition - p_fineP[j]->m_curPosition).norm() < searchArea)
	neighborIdx.push_back(j);
	}

	Vector3f start = m_boundary[i].m_minCorner + Vector3f(0.5f * m_lcp, 0.5f * m_lcp, 0.5f * m_lcp);
	float minDist = 300.0f;
	Vector3f minCandi;
	for (int x = 0; x < m_cntOfcandisPerOneAxis; x++)
	{
	float stepX = x*m_lcp;
	for (int y = 0; y < m_cntOfcandisPerOneAxis; y++)
	{
	float stepY = y*m_lcp;
	for (int z = 0; z < m_cntOfcandisPerOneAxis; z++)
	{
	Vector3f candi = start + Vector3f(stepX, stepY, z*m_lcp);

	bool nopeFlag = false;
	for (int k = 0; k < neighborIdx.size(); k++)
	{
	if ((candi - p_fineP[neighborIdx[k]]->m_curPosition).norm() < l0)
	{
	nopeFlag = true;
	break;
	}
	}
	for (int k = 0; k < m_boundary[i].m_boundaryNeighborList.size(); k++)
	{
	if ((candi - m_boundary[i].m_boundaryNeighborList[k]).norm() < l0)
	{
	nopeFlag = true;
	break;
	}
	}

	if (nopeFlag == false)
	{
	float dist = (candi - m_boundary[i].m_centerPosition).norm();
	if (dist < minDist)
	{
	minDist = dist;
	minCandi = candi;
	}
	}
	}
	}
	}

	if (minDist != 300.0f)
	{
	m_CPSet.push_back(minCandi);
	m_avgVelSet.push_back(m_boundary[i].m_avgVel);
	}
	*/
}

void FlowBoundary::NeighborSearchBTWTwoRes(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float searchRange = k.GetSmoothingRadius();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();

	for (int i = 0; i < fineP.size(); i++)
	{
		m_neighborListBTW[i].resize(0);
		Vector3f& finePos = fineP[i]->m_curPosition;

		for (int j = 0; j < coarseP.size(); j++)
		{
			Vector3f& coarsePos = coarseP[j]->m_curPosition;

			if ((finePos - coarsePos).norm() <= searchRange)
				m_neighborListBTW[i].push_back(j);
		}
	}
}

void FlowBoundary::UpdateAvgVelocity(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();

	// tempVel for coarseP
	float h = p_subWorld->GetTimeStep();
	for (int i = 0; i < coarseP.size(); i++)
	{
		if (coarseP[i]->m_mass != 0.0f)
		{
			m_tempVelforCoarse[i] = Vector3f(0.0f, -9.8f, 0.0f)*h + coarseP[i]->m_velocity;
		}
	}

	// MLS interpolation
	VectorXf acc_distVelX(10), acc_distVelY(10), acc_distVelZ(10);
	VectorXf res_distVelX(10), res_distVelY(10), res_distVelZ(10);
	VectorXf distFeature(10);
	MatrixXf acc_distMat(10, 10);

	for (int i = 0; i < fineP.size(); i++)
	{
		acc_distMat.setZero();
		acc_distVelX.setZero();
		acc_distVelY.setZero();
		acc_distVelZ.setZero();
		
		if (m_neighborListBTW[i].size() > 0)
		{
			Vector3f& finePos = fineP[i]->m_curPosition;

			for (int j = 0; j < m_neighborListBTW[i].size(); j++)
			{
				Vector3f& coarsePos = coarseP[m_neighborListBTW[i][j]]->m_curPosition;
				Vector3f dist = coarsePos - finePos;
				float weight = k.Cubic_Kernel(dist);

				distFeature[0] = 1;
				distFeature[1] = dist[0];
				distFeature[2] = dist[1];
				distFeature[3] = dist[2];
				distFeature[4] = dist[0] * dist[0];
				distFeature[5] = dist[0] * dist[1];
				distFeature[6] = dist[0] * dist[2];
				distFeature[7] = dist[1] * dist[1];
				distFeature[8] = dist[1] * dist[2];
				distFeature[9] = dist[2] * dist[2];

				for (int m = 0; m < 10; m++)
					for (int n = 0; n < 10; n++)
						acc_distMat(m, n) += weight * distFeature[m] * distFeature[n];

				for (int m = 0; m < 10; m++)
				{
					acc_distVelX[m] += weight * distFeature[m] * m_tempVelforCoarse[m_neighborListBTW[i][j]][0];
					acc_distVelY[m] += weight * distFeature[m] * m_tempVelforCoarse[m_neighborListBTW[i][j]][1];
					acc_distVelZ[m] += weight * distFeature[m] * m_tempVelforCoarse[m_neighborListBTW[i][j]][2];
				}
			}

			// velocity interpolation
			res_distVelX = acc_distMat.jacobiSvd(ComputeThinU | ComputeThinV).solve(acc_distVelX);
			res_distVelY = acc_distMat.jacobiSvd(ComputeThinU | ComputeThinV).solve(acc_distVelY);
			res_distVelZ = acc_distMat.jacobiSvd(ComputeThinU | ComputeThinV).solve(acc_distVelZ);
			//printf("(%d) res_distVel(%f, %f, %f)\n", i, res_distVelX[0], res_distVelY[0], res_distVelZ[0]);

			fineP[i]->m_velocity[0] = res_distVelX[0];
			fineP[i]->m_velocity[1] = res_distVelY[0];
			fineP[i]->m_velocity[2] = res_distVelZ[0];
		}
		else
			fineP[i]->m_velocity = Vector3f(0.0f, -9.8f, 0.0f)*h + fineP[i]->m_velocity;
		
	}
	
}

void FlowBoundary::InterpolateIISPH(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float h = p_subWorld->GetTimeStep();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();

	// MLS interpolation
	VectorXf acc_distVelX(10), acc_distVelY(10), acc_distVelZ(10), acc_density(10);
	VectorXf res_distVelX(10), res_distVelY(10), res_distVelZ(10), res_density(10);
	VectorXf distFeature(10);
	MatrixXf acc_distMat(10, 10);

	for (int i = 0; i < fineP.size(); i++)
	{
		acc_distMat.setZero();
		acc_distVelX.setZero();
		acc_distVelY.setZero();
		acc_distVelZ.setZero();

		acc_density.setZero();

		if (m_neighborListBTW[i].size() > 0)
		{
			Vector3f& finePos = fineP[i]->m_curPosition;

			for (int j = 0; j < m_neighborListBTW[i].size(); j++)
			{
				Vector3f& coarsePos = coarseP[m_neighborListBTW[i][j]]->m_curPosition;
				Vector3f dist = coarsePos - finePos;
				float weight = k.Cubic_Kernel(dist);

				distFeature[0] = 1;
				distFeature[1] = dist[0];
				distFeature[2] = dist[1];
				distFeature[3] = dist[2];
				distFeature[4] = dist[0] * dist[0];
				distFeature[5] = dist[0] * dist[1];
				distFeature[6] = dist[0] * dist[2];
				distFeature[7] = dist[1] * dist[1];
				distFeature[8] = dist[1] * dist[2];
				distFeature[9] = dist[2] * dist[2];

				for (int m = 0; m < 10; m++)
					for (int n = 0; n < 10; n++)
						acc_distMat(m, n) += weight * distFeature[m] * distFeature[n];

				for (int m = 0; m < 10; m++)
				{
					int idx = m_neighborListBTW[i][j];

					acc_distVelX[m] += weight * distFeature[m] * coarseP[idx]->m_velocity[0];
					acc_distVelY[m] += weight * distFeature[m] * coarseP[idx]->m_velocity[1];
					acc_distVelZ[m] += weight * distFeature[m] * coarseP[idx]->m_velocity[2];
					acc_density[m]  += weight * distFeature[m] * coarseP[idx]->m_density;
				}
			}

			// velocity interpolation
			res_distVelX = acc_distMat.jacobiSvd(ComputeThinU | ComputeThinV).solve(acc_distVelX);
			res_distVelY = acc_distMat.jacobiSvd(ComputeThinU | ComputeThinV).solve(acc_distVelY);
			res_distVelZ = acc_distMat.jacobiSvd(ComputeThinU | ComputeThinV).solve(acc_distVelZ);
			
			// pressure interpolation
			res_density  = acc_distMat.jacobiSvd(ComputeThinU | ComputeThinV).solve(acc_density);

			fineP[i]->m_velocity[0] = res_distVelX[0];
			fineP[i]->m_velocity[1] = res_distVelY[0];
			fineP[i]->m_velocity[2] = res_distVelZ[0];
			fineP[i]->m_density = res_density[0];
		}
		else
		{
			fineP[i]->m_velocity = Vector3f(0.0f, -9.8f, 0.0f) * h + fineP[i]->m_velocity;
			//fineP[i]->m_density = p_subWorld->GetRestDensity();
		}
	}
}