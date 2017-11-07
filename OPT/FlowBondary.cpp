#include "FlowBoundary.h"

void FlowBoundary::CreateFlowBoundary(Vector3f start, Vector3f end, float p_fineR)
{
	m_gridSize = 2.0f * p_fineR;
	m_lcp = m_gridSize * 0.01f;
	m_cntOfcandisPerOneAxis = int((m_gridSize - m_lcp) / m_lcp) + 1;

	CreateBoundaryWall(start, end);
	m_boundarySize = m_boundary.size();

	float coarseR = 2.0f * p_fineR;
	k.SetSmoothingRadius(4.0f * coarseR);
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

	m_neighborListBTWforFine.resize(p_subWorld->GetNumOfParticles());
	m_tempVelforCoarse.resize(p_mainWorld->GetNumOfParticles());
	
	m_neighborListBTWforCoarse.resize(p_mainWorld->GetNumOfParticles());
	m_tempVelforFine.resize(p_subWorld->GetNumOfParticles());

	// for PBFC
	m_PBFCData.m_lambdaForCoarse.resize(p_mainWorld->GetNumOfParticles());
	m_PBFCData.m_corrWithDensity.resize(p_subWorld->GetNumOfParticles());
	m_PBFCData.m_corrWithVelocity.resize(p_subWorld->GetNumOfParticles());
	m_PBFCData.m_weightForVelocityC.resize(p_subWorld->GetNumOfParticles());

	m_neighListwithFineP.resize(p_mainWorld->GetNumOfParticles());
	m_neighListwithBoundaryFineP.resize(p_mainWorld->GetNumOfParticles());
}

void FlowBoundary::NeighborSearchBTWTwoRes(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float searchRange = k.GetSmoothingRadius();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();

	for (int i = 0; i < fineP.size(); i++)
	{
		m_neighborListBTWforFine[i].clear();
		m_neighborListBTWforFine[i].resize(0);
		Vector3f& finePos = fineP[i]->m_curPosition;

		for (int j = 0; j < coarseP.size(); j++)
		{
			Vector3f& coarsePos = coarseP[j]->m_curPosition;

			if ((finePos - coarsePos).norm() <= searchRange)
				m_neighborListBTWforFine[i].push_back(j);
		}
	}
}
void FlowBoundary::NeighborSearchBTWTwoRes2(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float searchRange = k.GetSmoothingRadius();

	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();

	for (int i = 0; i < coarseP.size(); i++)
	{
		m_neighborListBTWforCoarse[i].clear();
		m_neighborListBTWforCoarse[i].resize(0);
	}

	for (int i = 0; i < coarseP.size(); i++)
	{
		Vector3f& coarsePos = coarseP[i]->m_curPosition;

		for (int j = 0; j < fineP.size(); j++)
		{
			Vector3f& finePos = fineP[j]->m_curPosition;

			if ((finePos - coarsePos).norm() <= searchRange)
				m_neighborListBTWforCoarse[i].push_back(j);
		}
	}
}

void FlowBoundary::InterpolateVelocity(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
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
		
		if (m_neighborListBTWforFine[i].size() > 0)
		{
			Vector3f& finePos = fineP[i]->m_curPosition;

			for (int j = 0; j < m_neighborListBTWforFine[i].size(); j++)
			{
				Vector3f& coarsePos = coarseP[m_neighborListBTWforFine[i][j]]->m_curPosition;
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
					acc_distVelX[m] += weight * distFeature[m] * m_tempVelforCoarse[m_neighborListBTWforFine[i][j]][0];
					acc_distVelY[m] += weight * distFeature[m] * m_tempVelforCoarse[m_neighborListBTWforFine[i][j]][1];
					acc_distVelZ[m] += weight * distFeature[m] * m_tempVelforCoarse[m_neighborListBTWforFine[i][j]][2];
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

		if (m_neighborListBTWforFine[i].size() > 0)
		{
			Vector3f& finePos = fineP[i]->m_curPosition;

			for (int j = 0; j < m_neighborListBTWforFine[i].size(); j++)
			{
				Vector3f& coarsePos = coarseP[m_neighborListBTWforFine[i][j]]->m_curPosition;
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
					int idx = m_neighborListBTWforFine[i][j];

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
			
			// density interpolation
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

void FlowBoundary::InterpolateWCSPH(FluidWorld* p_mainWorld, FluidWorld* p_subWorld, bool p_debugFlag)
{
	float h = p_subWorld->GetTimeStep();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	WCSPHWorld* mainSimulMethod = (WCSPHWorld*)p_mainWorld->GetFluidMethod();
	WCSPHWorld* subSimulMethod = (WCSPHWorld*)p_subWorld->GetFluidMethod();
	
	// tempVel for fineP
	for (int i = 0; i < coarseP.size(); i++)
	{
		if (coarseP[i]->m_mass != 0.0f)
		{
			m_tempVelforCoarse[i] = coarseP[i]->m_acceleration * h + coarseP[i]->m_velocity;
		}
	}

	// MLS interpolation
	VectorXf acc_distVelX(10), acc_distVelY(10), acc_distVelZ(10), acc_pressure(10);
	VectorXf res_distVelX(10), res_distVelY(10), res_distVelZ(10), res_pressure(10);
	VectorXf distFeature(10);
	MatrixXf acc_distMat(10, 10);

	for (int i = 0; i < fineP.size(); i++)
	{
		fineP[i]->m_interpolated = false;

		acc_distMat.setZero();
		acc_distVelX.setZero();
		acc_distVelY.setZero();
		acc_distVelZ.setZero();

		acc_pressure.setZero();

		if (m_neighborListBTWforFine[i].size() > 0)
		{
			Vector3f& finePos = fineP[i]->m_curPosition;

			for (int j = 0; j < m_neighborListBTWforFine[i].size(); j++)
			{
				int idx = m_neighborListBTWforFine[i][j];
				Vector3f& coarsePos = coarseP[idx]->m_curPosition;
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
					acc_distVelX[m] += weight * distFeature[m] * m_tempVelforCoarse[idx][0];
					acc_distVelY[m] += weight * distFeature[m] * m_tempVelforCoarse[idx][1];
					acc_distVelZ[m] += weight * distFeature[m] * m_tempVelforCoarse[idx][2];
					acc_pressure[m] += weight * distFeature[m] * mainSimulMethod->m_pressure[idx];
				}
			}

			// velocity interpolation
			res_distVelX = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_distVelX);
			res_distVelY = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_distVelY);
			res_distVelZ = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_distVelZ);
			
			fineP[i]->m_velocity[0] = res_distVelX[0];
			fineP[i]->m_velocity[1] = res_distVelY[0];
			fineP[i]->m_velocity[2] = res_distVelZ[0];

			// density interpolation
			res_pressure = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_pressure);
			res_pressure[0] > 0.0f ? subSimulMethod->m_pressure[i] = res_pressure[0] : subSimulMethod->m_pressure[i] = 0.0f;

			fineP[i]->m_interpolated = true;
		}
	}
}
void FlowBoundary::InterpolateWCSPH2(FluidWorld* p_mainWorld, FluidWorld* p_subWorld, bool p_debugFlag)
{
	float h = p_subWorld->GetTimeStep();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	WCSPHWorld* mainSimulMethod = (WCSPHWorld*)p_mainWorld->GetFluidMethod();
	WCSPHWorld* subSimulMethod = (WCSPHWorld*)p_subWorld->GetFluidMethod();

	// tempVel for fineP
	for (int i = 0; i < fineP.size(); i++)
	{
		if (fineP[i]->m_mass != 0.0f)
		{
			m_tempVelforFine[i] = fineP[i]->m_acceleration * h + fineP[i]->m_velocity;
		}
	}

	// MLS interpolation
	VectorXf acc_distVelX(10), acc_distVelY(10), acc_distVelZ(10), acc_pressure(10);
	VectorXf res_distVelX(10), res_distVelY(10), res_distVelZ(10), res_pressure(10);
	VectorXf distFeature(10);
	MatrixXf acc_distMat(10, 10);

	for (int i = 0; i < coarseP.size(); i++)
	{
		coarseP[i]->m_interpolated = false;

		acc_distMat.setZero();
		acc_distVelX.setZero();
		acc_distVelY.setZero();
		acc_distVelZ.setZero();

		acc_pressure.setZero();
		
		int neighborSize = m_neighborListBTWforCoarse[i].size();
		if (neighborSize > 0)
		{
			Vector3f& coarsePos = coarseP[i]->m_curPosition;

			for (int j = 0; j < neighborSize; j++)
			{
				int idx = m_neighborListBTWforCoarse[i][j];
				Vector3f& finePos = fineP[idx]->m_curPosition;
				Vector3f dist = finePos - coarsePos;
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
					acc_distVelX[m] += weight * distFeature[m] * m_tempVelforFine[idx][0];
					acc_distVelY[m] += weight * distFeature[m] * m_tempVelforFine[idx][1];
					acc_distVelZ[m] += weight * distFeature[m] * m_tempVelforFine[idx][2];
					acc_pressure[m] += weight * distFeature[m] * subSimulMethod->m_pressure[idx];
				}
			}

			// velocity interpolation
			res_distVelX = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_distVelX);
			res_distVelY = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_distVelY);
			res_distVelZ = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_distVelZ);

			coarseP[i]->m_velocity[0] = res_distVelX[0];
			coarseP[i]->m_velocity[1] = res_distVelY[0];
			coarseP[i]->m_velocity[2] = res_distVelZ[0];

			// pressure interpolation
			res_pressure = acc_distMat.jacobiSvd(ComputeFullU | ComputeFullV).solve(acc_pressure);
			res_pressure[0] > 0.0f ? mainSimulMethod->m_pressure[i] = res_pressure[0] : mainSimulMethod->m_pressure[i] = 0.0f;

			coarseP[i]->m_interpolated = true;
		}
	}
}

void FlowBoundary::NeighborBTWTwoResForPBFC(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float searchRange = 4.0f * p_subWorld->GetParticleRadius();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	std::vector<Vector3f>& boundaryFineP = p_subWorld->GetBoundaryParticleList();

	for (int i = 0; i < coarseP.size(); i++)
	{
		Vector3f& coarsePos = coarseP[i]->m_curPosition;

		m_neighListwithFineP[i].clear();
		m_neighListwithFineP[i].resize(0);
		for (int j = 0; j < fineP.size(); j++)
		{
			Vector3f& finePos = fineP[j]->m_curPosition;
			if ((coarsePos - finePos).norm() <= searchRange)
				m_neighListwithFineP[i].push_back(j);
		}

		m_neighListwithBoundaryFineP.clear();
		m_neighListwithBoundaryFineP[i].resize(0);
		for (int j = 0; j < boundaryFineP.size(); j++)
		{
			Vector3f& finePos = boundaryFineP[j];
			if ((coarsePos - finePos).norm() <= searchRange)
				m_neighListwithBoundaryFineP[i].push_back(j);
		}
	}
}
void FlowBoundary::SolvePBFCConstaints(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	int numOfCoarse = p_mainWorld->GetNumOfParticles();
	int numOfFine = p_subWorld->GetNumOfParticles();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	std::vector<Vector3f>& boundaryFineP = p_subWorld->GetBoundaryParticleList();

	float mainRestDensity = p_mainWorld->GetRestDensity();
	float intensityOfDensityC = 1.0f;
	float intensityOfVelosityC = 1.0f;

	for (int i = 0; i < numOfFine; i++)
	{
		m_PBFCData.m_corrWithDensity[i].setZero();
		m_PBFCData.m_corrWithVelocity[i].setZero();
		m_PBFCData.m_lambdaForCoarse[i] = 0.0f;
		m_PBFCData.m_weightForVelocityC[i] = 0.0f;

	}

	// update CoarseLambda & correction with Density Constraint
	for (int i = 0; i < numOfCoarse; i++)
	{
		Vector3f& coarsePos = coarseP[i]->m_curPosition;

		float density = 0; 
		for (int j = 0; j < m_neighListwithFineP[i].size(); j++)
		{
			int idx = m_neighListwithFineP[i][j];
			Vector3f& finePos = fineP[idx]->m_curPosition;
			density += fineP[idx]->m_mass * k.Cubic_Kernel(coarsePos-finePos);
		}
		for (int j = 0; j < m_neighListwithBoundaryFineP[i].size(); j++)
		{
			int idx = m_neighListwithBoundaryFineP[i][j];
			Vector3f& finePos = boundaryFineP[idx];
			density += p_subWorld->GetBoundaryPsi(idx) * k.Cubic_Kernel(coarsePos - finePos);
		}

		float C = std::max(density / mainRestDensity - 1.0f, 0.0f);

		if (C != 0.0f)
		{
			// Compute gradients dC/dx_j 
			float sum_grad_C2 = 0.0;
			Vector3f gradC_i(0.0f, 0.0f, 0.0f);

			for (unsigned int j = 0; j < m_neighListwithFineP[i].size(); j++)
			{
				unsigned int idx = m_neighListwithFineP[i][j];
				Vector3f& finePos = fineP[idx]->m_curPosition;

				Vector3f gradC_j = -fineP[idx]->m_mass / mainRestDensity * k.Cubic_Kernel_Gradient(coarsePos-finePos);
				sum_grad_C2 += gradC_j.squaredNorm();
				gradC_i -= gradC_j;
			}

			for (unsigned int j = 0; j < m_neighListwithBoundaryFineP[i].size(); j++)
			{
				int idx = m_neighListwithBoundaryFineP[i][j];
				Vector3f& finePos = boundaryFineP[idx];

				Vector3f gradC_j = -p_subWorld->GetBoundaryPsi(idx) / mainRestDensity * k.Cubic_Kernel_Gradient(coarsePos - finePos);
				sum_grad_C2 += gradC_j.squaredNorm();
				gradC_i -= gradC_j;
			}

			sum_grad_C2 += gradC_i.squaredNorm();

			// Compute lambda
			m_PBFCData.m_lambdaForCoarse[i] = -C / (sum_grad_C2 + 1.0e-6);
		}

		// calc correction with density constraint
		if (m_PBFCData.m_lambdaForCoarse[i] != 0.0f)
		{
			for (unsigned int j = 0; j < m_neighListwithFineP[i].size(); j++)
			{
				unsigned int idx = m_neighListwithFineP[i][j];
				Vector3f& finePos = fineP[idx]->m_curPosition;

				Vector3f gradC_j = -fineP[idx]->m_mass / mainRestDensity * k.Cubic_Kernel_Gradient(coarsePos - finePos);
				m_PBFCData.m_corrWithDensity[idx] += intensityOfDensityC * m_PBFCData.m_lambdaForCoarse[i] * gradC_j;
			}
		}

		// calc correction with Velocity Constraint1
		for (unsigned int j = 0; j < m_neighListwithFineP[i].size(); j++)
		{
			unsigned int idx = m_neighListwithFineP[i][j];
			Vector3f& finePos = fineP[idx]->m_curPosition;

			m_PBFCData.m_corrWithVelocity[idx] += coarseP[i]->m_velocity * k.Cubic_Kernel(finePos - coarsePos);
			m_PBFCData.m_weightForVelocityC[idx] += k.Cubic_Kernel(finePos - coarsePos);
		}
	}
	
	// calc correction with Velocity Constraint2
	for (int i = 0; i < numOfFine; i++)
	{
		if (m_PBFCData.m_corrWithVelocity[i].norm() != 0.0f)
		{
			m_PBFCData.m_corrWithVelocity[i] = m_PBFCData.m_corrWithVelocity[i] / m_PBFCData.m_weightForVelocityC[i];
			m_PBFCData.m_corrWithVelocity[i] = intensityOfVelosityC * p_subWorld->GetTimeStep()
				* (m_PBFCData.m_corrWithVelocity[i] - fineP[i]->m_velocity);
		}
		
		fineP[i]->m_curPosition += m_PBFCData.m_corrWithVelocity[i] + m_PBFCData.m_corrWithDensity[i];
	}

}