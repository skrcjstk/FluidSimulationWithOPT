#include "PBFControl.h"

void PBFControl::Initialize(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float mainR = p_mainWorld->GetParticleRadius();
	constKernel.SetSmoothingRadius(4.0f * mainR);

	m_PBFCData.m_lambdaForCoarse.resize(p_mainWorld->GetNumOfParticles());
	m_PBFCData.m_corrWithDensity.resize(p_subWorld->GetNumOfParticles());
	m_PBFCData.m_corrWithVelocity.resize(p_subWorld->GetNumOfParticles());
	m_PBFCData.m_weightForVelocityC.resize(p_subWorld->GetNumOfParticles());

	m_neighListwithSubP.resize(p_mainWorld->GetNumOfParticles());
	m_neighListwithSubBoundaryP.resize(p_mainWorld->GetNumOfParticles());

	m_tDataForMain.resize(p_subWorld->GetNumOfParticles());
	//m_tDataForMainBoundary.resize(p_subWorld->GetNumOfParticles());
	m_tDataForSub.resize(p_subWorld->GetNumOfParticles());
	m_deltaPWithControl.resize(p_subWorld->GetNumOfParticles());
	for (int i = 0; i < p_subWorld->GetNumOfParticles(); i++)
	{
		m_deltaPWithControl[i] = Vector3f(0.0f, 0.0f, 0.0f);
	}
}

void PBFControl::NeighborBTWTwoResForPBFC(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float searchRange = constKernel.GetSmoothingRadius();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	std::vector<Vector3f>& boundaryFineP = p_subWorld->GetBoundaryParticleList();
	std::vector<Vector3f>& boundaryCoarseP = p_mainWorld->GetBoundaryParticleList();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < coarseP.size(); i++)
		{
			Vector3f& coarsePos = coarseP[i]->m_curPosition;

			m_neighListwithSubP[i].clear();
			m_neighListwithSubP[i].resize(0);
			for (int j = 0; j < fineP.size(); j++)
			{
				Vector3f& finePos = fineP[j]->m_curPosition;
				Vector3f r = coarsePos - finePos;
				if (r.norm() <= searchRange)
				{
					m_neighListwithSubP[i].push_back(j);
				}
			}

			m_neighListwithSubBoundaryP[i].clear();
			m_neighListwithSubBoundaryP[i].resize(0);
			for (int j = 0; j < boundaryFineP.size(); j++)
			{
				Vector3f& finePos = boundaryFineP[j];
				if ((coarsePos - finePos).norm() <= searchRange)
					m_neighListwithSubBoundaryP[i].push_back(j);
			}
		}
	}

}
void PBFControl::SolvePBFCConstaints(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	int numOfCoarse = p_mainWorld->GetNumOfParticles();
	int numOfFine = p_subWorld->GetNumOfParticles();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	std::vector<Vector3f>& boundaryFineP = p_subWorld->GetBoundaryParticleList();

	float mainRestDensity = p_mainWorld->GetRestDensity();


#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < numOfFine; i++)
		{
			m_PBFCData.m_corrWithDensity[i].setZero();
			m_PBFCData.m_corrWithVelocity[i].setZero();
			m_PBFCData.m_weightForVelocityC[i] = 0.0f;
		}

		// update CoarseLambda & correction with Density Constraint
#pragma omp for schedule(static) 
		for (int i = 0; i < numOfCoarse; i++)
		{
			m_PBFCData.m_lambdaForCoarse[i] = 0.0f;
			Vector3f& coarsePos = coarseP[i]->m_curPosition;

			float density = 0;
			for (int j = 0; j < m_neighListwithSubP[i].size(); j++)
			{
				int idx = m_neighListwithSubP[i][j];
				Vector3f& finePos = fineP[idx]->m_curPosition;
				density += fineP[idx]->m_mass * constKernel.Cubic_Kernel(coarsePos - finePos);
			}
			for (int j = 0; j < m_neighListwithSubBoundaryP[i].size(); j++)
			{
				int idx = m_neighListwithSubBoundaryP[i][j];
				Vector3f& finePos = boundaryFineP[idx];
				density += p_subWorld->GetBoundaryPsi(idx) * constKernel.Cubic_Kernel(coarsePos - finePos);
			}

			float C = std::max(density / mainRestDensity - 1.0f, 0.0f);

			if (C != 0.0f)
			{
				// Compute gradients dC/dx_j 
				float sum_grad_C2 = 0.0;
				Vector3f gradC_i(0.0f, 0.0f, 0.0f);

				for (int j = 0; j < m_neighListwithSubP[i].size(); j++)
				{
					int idx = m_neighListwithSubP[i][j];
					Vector3f& finePos = fineP[idx]->m_curPosition;

					Vector3f gradC_j = -fineP[idx]->m_mass / mainRestDensity * constKernel.Cubic_Kernel_Gradient(coarsePos - finePos);
					sum_grad_C2 += gradC_j.squaredNorm();
					gradC_i -= gradC_j;
				}

				for (int j = 0; j < m_neighListwithSubBoundaryP[i].size(); j++)
				{
					int idx = m_neighListwithSubBoundaryP[i][j];
					Vector3f& finePos = boundaryFineP[idx];

					Vector3f gradC_j = -p_subWorld->GetBoundaryPsi(idx) / mainRestDensity * constKernel.Cubic_Kernel_Gradient(coarsePos - finePos);
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
				for (int j = 0; j < m_neighListwithSubP[i].size(); j++)
				{
					int idx = m_neighListwithSubP[i][j];
					Vector3f& finePos = fineP[idx]->m_curPosition;

					Vector3f gradC_j = -fineP[idx]->m_mass / mainRestDensity * constKernel.Cubic_Kernel_Gradient(coarsePos - finePos);
					m_PBFCData.m_corrWithDensity[idx] += m_intensityOfDensityC * m_PBFCData.m_lambdaForCoarse[i] * gradC_j;
				}
			}

			// calc correction with Velocity Constraint1
			for (int j = 0; j < m_neighListwithSubP[i].size(); j++)
			{
				int idx = m_neighListwithSubP[i][j];
				Vector3f& finePos = fineP[idx]->m_curPosition;

				m_PBFCData.m_corrWithVelocity[idx] += coarseP[i]->m_velocity * constKernel.Cubic_Kernel(finePos - coarsePos);
				m_PBFCData.m_weightForVelocityC[idx] += constKernel.Cubic_Kernel(finePos - coarsePos);
			}
		}

#pragma omp for schedule(static) 
		// calc correction with Velocity Constraint2
		for (int i = 0; i < numOfFine; i++)
		{
			if (m_PBFCData.m_corrWithVelocity[i].norm() != 0.0f)
			{
				m_PBFCData.m_corrWithVelocity[i] = m_PBFCData.m_corrWithVelocity[i] / m_PBFCData.m_weightForVelocityC[i];
				m_PBFCData.m_corrWithVelocity[i] = m_intensityOfVelocityC * p_subWorld->GetTimeStep()
					* (m_PBFCData.m_corrWithVelocity[i] - fineP[i]->m_velocity);
			}

			fineP[i]->m_curPosition += m_PBFCData.m_corrWithDensity[i] + m_PBFCData.m_corrWithVelocity[i];
			m_deltaPWithControl[i] = m_PBFCData.m_corrWithVelocity[i] + m_PBFCData.m_corrWithDensity[i];
		}
	}
}
void PBFControl::UpdateTrainingDataForMain(FluidWorld* p_mainWorld, FluidWorld* p_subWorld)
{
	float searchRange = constKernel.GetSmoothingRadius();
	std::vector<FParticle*>& coarseP = p_mainWorld->GetParticleList();
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	std::vector<Vector3f>& boundaryCoarseP = p_mainWorld->GetBoundaryParticleList();

	for (int i = 0; i < fineP.size(); i++)
	{
		m_tDataForMain[i].clear();
		m_tDataForMain[i].resize(0);
	}

	for (int i = 0; i < coarseP.size(); i++)
	{
		Vector3f& coarsePos = coarseP[i]->m_curPosition;
		for (int j = 0; j < fineP.size(); j++)
		{
			Vector3f& finePos = fineP[j]->m_curPosition;
			Vector3f r = coarsePos - finePos;
			if (r.norm() <= searchRange)
			{
				TrainData a;
				a.mass = coarseP[i]->m_mass;
				a.kWeight = constKernel.Cubic_Kernel(r);
				a.RVec = r;
				a.kGrad = constKernel.Cubic_Kernel_Gradient(r);
				a.dPos = coarseP[i]->m_curPosition - coarseP[i]->m_oldPosition;
				m_tDataForMain[j].push_back(a);
			}
		}
	}

	for (int i = 0; i < boundaryCoarseP.size(); i++)
	{
		float psi = p_mainWorld->GetBoundaryPsi(i);
		for (int j = 0; j < fineP.size(); j++)
		{
			Vector3f& finePos = fineP[j]->m_curPosition;
			Vector3f r = boundaryCoarseP[i] - finePos;
			if (r.norm() <= searchRange)
			{
				TrainData a;
				a.mass = psi;
				a.kWeight = constKernel.Cubic_Kernel(r);
				a.RVec = r;
				a.kGrad = constKernel.Cubic_Kernel_Gradient(r);
				a.dPos.setZero();
				m_tDataForMain[j].push_back(a);
			}
		}
	}
}

void PBFControl::UpdateTrainingDataForSub(FluidWorld* p_subWorld)
{
	std::vector<FParticle*>& fineP = p_subWorld->GetParticleList();
	std::vector<Vector3f>& boundaryFineP = p_subWorld->GetBoundaryParticleList();
	FluidKernel& subKernel = p_subWorld->GetKernel();

	for (int i = 0; i < fineP.size(); i++)
	{
		m_tDataForSub[i].clear();
		m_tDataForSub[i].resize(0);

		Vector3f& finePos = fineP[i]->m_curPosition;

		for (int j = 0; j < fineP[i]->m_neighborList.size(); j++)
		{
			TrainData a;
			unsigned int idx = fineP[i]->m_neighborList[j];
			Vector3f r = finePos - fineP[idx]->m_curPosition;
			a.mass = fineP[idx]->m_mass;
			a.kWeight = subKernel.Cubic_Kernel(r);
			a.RVec = r;
			a.kGrad = subKernel.Cubic_Kernel_Gradient(r);
			a.dPos = fineP[idx]->m_curPosition - fineP[idx]->m_oldPosition;
			m_tDataForSub[i].push_back(a);
		}
		/*
		for (int j = 0; j < fineP[i]->m_neighborBoundaryList.size(); j++)
		{
			TrainData a;
			unsigned int idx = fineP[i]->m_neighborBoundaryList[j];
			Vector3f r = finePos - boundaryFineP[idx];
			a.mass = p_subWorld->GetBoundaryPsi(idx);
			//a.kWeight = subKernel.Cubic_Kernel(r);
			a.RVec = r;
			//a.kGrad = subKernel.Cubic_Kernel_Gradient(r);
			m_tDataForSub[i].push_back(a);
		}
		*/
	}
}