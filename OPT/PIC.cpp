#include "PIC.h"

void PIC::Initialize(FluidWorld* p_world, Vector3f p_origin, Vector3f p_bSize, Vector3i p_nCount, float p_rho)
{
	int np = p_world->GetNumOfParticles();
	AssignResultF.resize(np);

	rho = p_rho;
	origin = p_origin;
	
	ni = p_nCount[0];
	nj = p_nCount[1];
	nk = p_nCount[2];
	dx = p_bSize[0] / (float)ni;
	dy = p_bSize[1] / (float)nj;
	dz = p_bSize[2] / (float)nk;
	
	u = APICArray3d::Array3d<float>(ni + 1, nj, nk, 0.0f);
	v = APICArray3d::Array3d<float>(ni, nj + 1, nk, 0.0f);
	w = APICArray3d::Array3d<float>(ni, nj, nk + 1, 0.0f);

	geo.resize(nk*nj*ni);
	cellsForF.resize(nk*nj*ni);
	cellsForB.resize(nk*nj*ni);
	cells_centor_pos.resize(nk*nj*ni);
	cells_centor_uvw_coord.resize(nk*nj*ni);

	for (int k = 0; k < nk; k++)
		for (int j = 0; j < nj; j++)
			for (int i = 0; i < ni; i++)
			{
				int idx = (k*nj*ni) + j*ni + i;
				Vector3f uvw_coord = Vector3f((i + 0.5f) * dx, (j + 0.5f) * dy, (k + 0.5f) * dz);
				cells_centor_pos[idx] = origin + uvw_coord;
				cells_centor_uvw_coord[idx] = Vector3f(uvw_coord[0] / dx, uvw_coord[1] / dy, uvw_coord[2] / dz);
				geo[idx] = A;
			}

	printf("APIC Grid(%d, %d, %d)\n", ni, nj, nk);
	printf("APIC dx(%.2f) dy(%.2f) dz(%.2f)\n", dx, dy, dz);
}

void PIC::clean()
{
	u.~Array3d();
	v.~Array3d();
	w.~Array3d();

	geo.clear();
	cellsForF.clear();
	cellsForB.clear();
	cells_centor_pos.clear();
	cells_centor_uvw_coord.clear();
	AssignResultF.clear();
}

void PIC::AssignBoundary(std::vector<FParticle*>& p_BpList)
{
	int nBp = (int)p_BpList.size();
	for (int n = 0; n < nBp; n++)
	{
		FParticle* p = p_BpList[n];
		
		int pi = (int)round((p->m_curPosition[0] - origin[0]) / dx);
		int pj = (int)round((p->m_curPosition[1] - origin[1]) / dy);
		int pk = (int)round((p->m_curPosition[2] - origin[2]) / dz);
		int i = pi >= 0 && pi < ni ? pi : -1;
		int j = pj >= 0 && pj < nj ? pj : -1;
		int k = pk >= 0 && pk < nk ? pk : -1;
		
		int idx = (k*nj*ni) + j*ni + i;
		cellsForB[idx].push_back(p);

		geo[idx] = B;
		
	}
}

void PIC::AssignCells(FluidWorld* p_world)
{
	int np = p_world->GetNumOfParticles();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)
		for (int k = 0; k < nk; k++)
			for (int j = 0; j < nj; ++j)
				for (int i = 0; i < ni; ++i)
				{
					int idx = (k*nj*ni) + j*ni + i;
					cellsForF[idx].clear();
					
					// marking as air
					if(geo[idx] == F)
						geo[idx] = A;
				}

#pragma omp for schedule(static)
		for (int n = 0; n < np; n++)
		{
			FParticle *p = p_world->GetParticle(n);

			int pi = (int)round((p->m_curPosition[0] - origin[0]) / dx);
			int pj = (int)round((p->m_curPosition[1] - origin[1]) / dy);
			int pk = (int)round((p->m_curPosition[2] - origin[2]) / dz);
			int i = pi >= 0 && pi < ni ? pi : -1;
			int j = pj >= 0 && pj < nj ? pj : -1;
			int k = pk >= 0 && pk < nk ? pk : -1;

			AssignResultF[n][0] = i;
			AssignResultF[n][1] = j;
			AssignResultF[n][2] = k;
		}
	}

	for (int n = 0; n < np; n++)
	{
		FParticle *p = p_world->GetParticle(n);
		Vector3i& assign = AssignResultF[n];
		if (assign[0] != -1 && assign[1] != -1 && assign[2] != -1)
		{
			int idx = (assign[2] * nj*ni) + assign[1] * ni + assign[0];
			cellsForF[idx].push_back(p);
			
			// marking as fluid
			if (geo[idx] != B)
				geo[idx] = F;
		}
	}
}

void PIC::Map_P2G(FluidWorld* p_world)
{
	float radii = p_world->GetParticleRadius();

#pragma omp parallel default(shared)
	{
		// u-component of velocity
#pragma omp for schedule(static)
		for (int k = 0; k < nk; k++)
			for (int j = 0; j < nj; j++)
				for (int i = 0; i < ni + 1; i++)
				{
					std::vector<FParticle *> neighbors;
					GetNeigboringParticles_cell(i, j, k, -1, 0, -1, 1, -1, 1, neighbors);
					
					if ((int)neighbors.size() > 0)
					{
						Vector3f pos = Vector3f(i * dx, (j + 0.5f)*dy, (k + 0.5f)*dz) + origin;

						float sum_weight = 0.0;
						float sum_u = 0.0;
						for (FParticle* p : neighbors)
						{
							float weight = 4.0f / 3.0f * M_PI * rho * radii * radii * radii * linear_kernel(p->m_curPosition - pos, dx);
							sum_u += weight * p->m_velocity[0]; // +p->m_c.col(0).dot(pos - p->m_curPosition);
							sum_weight += weight;
						}

						if (sum_weight != 0.0)
							u.set(i, j, k, sum_u / sum_weight);
						else
							u.set(i, j, k, 0.0);
					}
					else
						u.set(i, j, k, 0.0);
				}

		// v-component of velocity
#pragma omp for schedule(static)
		for (int k = 0; k < nk; k++)
			for (int j = 0; j < nj + 1; j++)
				for (int i = 0; i < ni; i++)
				{
					std::vector<FParticle *> neighbors;
					GetNeigboringParticles_cell(i, j, k, -1, 1, -1, 0, -1, 1, neighbors);

					if ((int)neighbors.size() > 0)
					{
						Vector3f pos = Vector3f((i + 0.5)*dx, j * dy, (k + 0.5)*dz) + origin;

						float sum_weight = 0.0;
						float sum_u = 0.0;
						for (FParticle* p : neighbors)
						{
							float weight = 4.0f / 3.0f * M_PI * rho * radii * radii * radii * linear_kernel(p->m_curPosition - pos, dy);
							sum_u += weight * p->m_velocity[1]; //+ p->m_c.col(1).dot(pos - p->m_curPosition);
							sum_weight += weight;
						}

						if (sum_weight != 0.0)
							v.set(i, j, k, sum_u / sum_weight);
						else
							v.set(i, j, k, 0.0);
					}
					else
						v.set(i, j, k, 0.0);
				}

		// w-component of velocity
#pragma omp for schedule(static)
		for (int k = 0; k < nk + 1; k++)
			for (int j = 0; j < nj; j++)
				for (int i = 0; i < ni; i++)
				{
					std::vector<FParticle *> neighbors;
					GetNeigboringParticles_cell(i, j, k, -1, 1, -1, 1, -1, 0, neighbors);
					
					if ((int)neighbors.size() > 0)
					{
						float sum_weight = 0.0;
						float sum_u = 0.0;
						Vector3f pos = Vector3f((i + 0.5)*dx, (j + 0.5)*dy, k * dz) + origin;

						for (FParticle* p : neighbors)
						{
							float weight = 4.0f / 3.0f * M_PI * rho * radii * radii * radii * linear_kernel(p->m_curPosition - pos, dz);
							sum_u += weight * p->m_velocity[2]; // +p->m_c.col(2).dot(pos - p->m_curPosition);
							sum_weight += weight;
						}

						if (sum_weight != 0.0)
							w.set(i, j, k, sum_u / sum_weight);
						else
							w.set(i, j, k, 0.0);
					}
					else
						w.set(i, j, k, 0.0);
				}
	}
}

void PIC::GetNeigboringParticles_cell(int i, int j, int k, int wl, int wh, int hl, int hh, int dl, int dh, std::vector<FParticle *>& res)
{
	for (int sk = k + dl; sk <= k + dh; sk++)
		for (int sj = j + hl; sj <= j + hh; sj++)
			for (int si = i + wl; si <= i + wh; si++)
			{
				if (si < 0 || si > ni - 1 || sj < 0 || sj > nj - 1 || sk < 0 || sk > nk -1) 
					continue;
				res.insert(res.end(), cellsForB[(sk * nj * ni) + (sj * ni) + si].begin(), cellsForB[(sk * nj * ni) + (sj * ni) + si].end());
				res.insert(res.end(), cellsForF[(sk * nj * ni) + (sj * ni) + si].begin(), cellsForF[(sk * nj * ni) + (sj * ni) + si].end());
			}
}

Vector3f PIC::GetVelocity(int p_gridIdx)
{
	//Interpolate the velocity from the u and v grids
	Vector3f& uvw_coord = cells_centor_uvw_coord[p_gridIdx];
	Vector3f p0 = uvw_coord - Vector3f(0, 0.5, 0.5);
	Vector3f p1 = uvw_coord - Vector3f(0.5, 0, 0.5);
	Vector3f p2 = uvw_coord - Vector3f(0.5, 0.5, 0);
	float u_value = interpolate_value(p0, u);
	float v_value = interpolate_value(p1, v);
	float w_value = interpolate_value(p2, w);

	return Vector3f(u_value, v_value, w_value);
}

void PIC::GetDescriptorAll(float result[], int desc_width)
{
	int d = desc_width;
	int halfCnt = (int)(desc_width / 2);
	int dataSize = 4 * (d*d*d);
	int np = (int)AssignResultF.size();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)
		// m,u,v,w
		for (int n = 0; n < np; n++)
		{
			Vector3i& ijk = AssignResultF[n];
			int startidx = n * dataSize;

			for (int k = -halfCnt; k <= halfCnt; k++)
				for (int j = -halfCnt; j <= halfCnt; j++)
					for (int i = -halfCnt; i <= halfCnt; i++)
					{
						int idx = 4 * ( (k + halfCnt) * (d*d) + (j + halfCnt)*(d) + (i + halfCnt));
						
						Vector3i neiGrid = ijk + Vector3i(i, j, k);
						int neiIdx = k*nj*ni + j*ni + i;
						Vector3f vel = GetVelocity(neiIdx);

						result[startidx + idx + 0] = geo[neiIdx];
						result[startidx + idx + 1] = vel[0];
						result[startidx + idx + 2] = vel[1];
						result[startidx + idx + 3] = vel[2];
					}
		}
	}
}
void PIC::GetDescriptor(Vector3i p_ijk, float result[], int desc_width)
{
	int d = desc_width;
	int halfCnt = (int)(desc_width / 2);

	// m,u,v,w
	for (int k = -halfCnt; k <= halfCnt; k++)
		for (int j = -halfCnt; j <= halfCnt; j++)
			for (int i = -halfCnt; i <= halfCnt; i++)
			{
				int idx = 4 * ((k + halfCnt) * (d*d) + (j + halfCnt)*(d)+(i + halfCnt));

				Vector3i neiGrid = p_ijk + Vector3i(i, j, k);
				int neiIdx = k*nj*ni + j*ni + i;
				Vector3f vel = GetVelocity(neiIdx);

				result[idx + 0] = geo[neiIdx];
				result[idx + 1] = vel[0];
				result[idx + 2] = vel[1];
				result[idx + 3] = vel[2];
			}
	
}
inline float PIC::interpolate_value(Vector3f& point, APICArray3d::Array3d<float>& grid)
{
	int i, j, k;
	float fx, fy, fz;
	float result;

	get_barycentric(point[0], i, fx, 0, grid.width);
	get_barycentric(point[1], j, fy, 0, grid.height);
	get_barycentric(point[2], k, fz, 0, grid.depth);

	return trilerp(grid.get(i, j, k), grid.get(i + 1, j, k), grid.get(i, j + 1, k), grid.get(i + 1, j + 1, k),
		grid.get(i, j, k + 1), grid.get(i + 1, j, k + 1), grid.get(i, j + 1, k + 1), grid.get(i + 1, j + 1, k + 1),
		fx, fy, fz);

}
inline Vector3f PIC::affine_interpolate_value(Vector3f& point, APICArray3d::Array3d<float>& grid)
{
	int i, j, k;
	float fx, fy, fz;
	Vector3f result;

	get_barycentric(point[0], i, fx, 0, grid.width);
	get_barycentric(point[1], j, fy, 0, grid.height);
	get_barycentric(point[2], k, fz, 0, grid.depth);

	result = grad_trilerp(grid.get(i, j, k), grid.get(i + 1, j, k), grid.get(i, j + 1, k), grid.get(i + 1, j + 1, k), grid.get(i, j, k + 1), grid.get(i + 1, j, k + 1), grid.get(i, j + 1, k + 1), grid.get(i + 1, j + 1, k + 1), fx, fy, fz);

	return result;

}
void PIC::UpdateAffineMatrix(FluidWorld* p_world)
{
	for (int i = 0; i < p_world->GetNumOfParticles(); i++)
	{
		FParticle* particle = p_world->GetParticle(i);
		Vector3f dist = (particle->m_curPosition - origin);
		Vector3f p(dist[0] / dx, dist[1] / dy, dist[2] / dx);
		Vector3f p0 = p - Vector3f(0, 0.5, 0.5);
		Vector3f p1 = p - Vector3f(0.5, 0, 0.5);
		Vector3f p2 = p - Vector3f(0.5, 0.5, 0);

		particle->m_c.col(0) = affine_interpolate_value(p0, u) / dx;
		particle->m_c.col(1) = affine_interpolate_value(p1, v) / dy;
		particle->m_c.col(2) = affine_interpolate_value(p2, w) / dz;
	}

}
