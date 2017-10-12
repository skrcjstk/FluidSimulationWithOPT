#include "FlowBoundary.h"

void FlowBoundary::CreateFlowBoundary(Vector3f start, Vector3f end, float gridSize)
{
	m_gridSize = gridSize;
	m_lcp = m_gridSize * 0.01f;
	m_cntOfcandisPerOneAxis = int((m_gridSize - m_lcp) / m_lcp) + 1;

	CreateBoundaryWall(start, end);
	m_boundarySize = m_boundary.size();

	
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

	printf("diff(%f, %f, %f)\n", diff[0], diff[1], diff[2]);
	printf("count(%d, %d, %d)\n", countX, countY, countZ);

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

void FlowBoundary::SearchBoundaryNeighbor(std::vector<Vector3f>& p_boundaryParticles, float p_radius)
{
	for (int i = 0; i < m_boundarySize; i++)
	{
		m_boundary[i].m_boundaryNeighborList.resize(0);

		for (int j = 0; j < p_boundaryParticles.size(); j++)
		{
			float dist = (p_boundaryParticles[j] - m_boundary[i].m_centerPosition).norm();
			if (dist < 3.0f * p_radius)
			{
				m_boundary[i].m_boundaryNeighborList.push_back(Vector3f(p_boundaryParticles[j][0], p_boundaryParticles[j][1], p_boundaryParticles[j][2]));
			}
		}
	}
}

void FlowBoundary::CreateFinePs(std::vector<FParticle*>& p_coarseP, float p_coarseP_radius,
	std::vector<FParticle*>& p_fineP, float p_fineP_radius)
{
	std::vector<Vector3f> neighborCoarseParticles;
	float R = p_coarseP_radius;
	float r = p_fineP_radius;

	float Vcell = m_gridSize * m_gridSize * m_gridSize;

	for (int i = 0; i < m_boundarySize; i++)
		m_boundary[i].m_inFluid = false;
		
	for (int j = 0; j < p_coarseP.size(); j++) {
		
		for (int i = 0; i < m_boundarySize; i++) {
			if (m_boundary[i].m_inFluid == false)
			{
				float m_fluidRatio = 0.0f;
				float dist = (p_coarseP[j]->m_curPosition - m_boundary[i].m_centerPosition).norm();
				if (dist <= R)
				{
					// overlap ratio
					bool isOverlap = true;
					Vector3f length;
					Vector3f Vl = m_boundary[i].m_minCorner - p_coarseP[j]->m_curPosition;
					Vector3f Vr = m_boundary[i].m_maxCorner - p_coarseP[j]->m_curPosition;

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
					m_fluidRatio = length[0] * length[1] * length[2] / Vcell;

					if (isOverlap == true && m_fluidRatio >= m_fThr)
					{
						m_boundary[i].m_inFluid = true;
						m_boundary[i].m_avgVel = p_coarseP[j]->m_velocity;
						break;
					}
				}
			}
		}
	}
	
	m_avgVelSet.clear();
	m_avgVelSet.resize(0);
	m_CPSet.clear();
	m_CPSet.resize(0);

	float l0 = 2.0f * p_fineP_radius;

	for (int i = 0; i < m_boundarySize; i++)
	{
		if (m_boundary[i].m_inFluid)
		{
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
				
		}
	}
}
