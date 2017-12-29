#include "LocalAPICSim.h"

void LAPIC::Initialize(Vector3f p_bSize, Vector3i p_nCount, float p_rho)
{
	rho = p_rho;

	ni = p_nCount[0];
	nj = p_nCount[1];
	nk = p_nCount[2];

	int h_ni = (int)(ni / 2);
	int h_nj = (int)(nj / 2);
	int h_nk = (int)(nk / 2);

	dx = p_bSize[0] / (float)ni;
	dy = p_bSize[1] / (float)nj;
	dz = p_bSize[2] / (float)nk;

	centerIdx = (h_ni * nj*ni) + h_nj * ni + h_nk;

	float h_dx = 0.5f * dx;
	float h_dy = 0.5f * dy;
	float h_dz = 0.5f * dz;

	toOrigin[0] = -(h_dx + dx * h_ni);
	toOrigin[1] = -(h_dy + dy * h_nj);
	toOrigin[2] = -(h_dz + dz * h_nk);

	m = APICArray3d::Array3d<float>(ni, nj, nk, 0.0f);
	u = APICArray3d::Array3d<float>(ni + 1, nj, nk, 0.0f);
	v = APICArray3d::Array3d<float>(ni, nj + 1, nk, 0.0f);
	w = APICArray3d::Array3d<float>(ni, nj, nk + 1, 0.0f);

	cells.resize(nk*nj*ni);
	cells_pos.resize(nk*nj*ni);
	
	printf("APIC Grid(%d, %d, %d)\n", ni, nj, nk);
	printf("APIC dx(%.2f) dy(%.2f) dz(%.2f)\n", dx, dy, dz);
}

void LAPIC::LAPICDesc(float result[], FParticle* p_center, std::vector<FParticle*>& p_list, float p_radii)
{
	// Assign Cells
	origin = p_center->m_curPosition + toOrigin;
	cells[centerIdx].push_back(p_center);

	int np = p_list.size();
	AssignResult.resize(np);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)
	for (int k = 0; k < nk; k++)
		for (int j = 0; j < nj; j++)
			for (int i = 0; i < ni + 1; i++)
			{
				cells[(k*nj*ni) + j*ni + i].clear();
				cells_pos[(k*nj*ni) + j*ni + i] = Vector3f(i * dx, j * dy, k * dz) + origin;
			}

#pragma omp for schedule(static)
		for (int n = 0; n < p_list.size(); n++)
		{
			FParticle* p = p_list[n];

			int pi = (int)((p->m_curPosition[0] - origin[0]) / dx);
			int pj = (int)((p->m_curPosition[1] - origin[1]) / dy);
			int pk = (int)((p->m_curPosition[2] - origin[2]) / dz);
			AssignResult[n][0] = pi;
			AssignResult[n][1] = pj;
			AssignResult[n][2] = pk;
		}
	}

	for (int n = 0; n < np; n++)
	{
		Vector3i& assign = AssignResult[n];
		cells[(assign[2] * nj*ni) + assign[1] * ni + assign[0]].push_back(p_list[n]);
	}

	// Map Particles to Grid
#pragma omp parallel default(shared)
	{
		// u-component of velocity
#pragma omp for schedule(static)
		for (int k = 0; k < nk; k++)
			for (int j = 0; j < nj; j++)
				for (int i = 0; i < ni + 1; i++)
				{
					Vector3f pos = Vector3f(i * dx, (j + 0.5)*dy, (k + 0.5)*dz) + origin;
					std::vector<FParticle *> neighbors;
					GetNeigboringParticles_cell(i, j, k, -1, 0, -1, 1, -1, 1, neighbors);

					float sum_weight = 0.0;
					float sum_u = 0.0;
					for (FParticle* p : neighbors)
					{
						float weight = 4.0 / 3.0 * M_PI * rho * p_radii * p_radii * p_radii * linear_kernel(p->m_curPosition - pos, dx);
						sum_u += weight * p->m_velocity[0]; // +p->m_c.col(0).dot(pos - p->m_curPosition);
						sum_weight += weight;
					}

					if (sum_weight != 0.0)
						u.set(i, j, k, sum_u / sum_weight);
					else
						u.set(i, j, k, 0.0);
				}

		// v-component of velocity
#pragma omp for schedule(static)
		for (int k = 0; k < nk; k++)
			for (int j = 0; j < nj + 1; j++)
				for (int i = 0; i < ni; i++)
				{
					Vector3f pos = Vector3f((i + 0.5)*dx, j * dy, (k + 0.5)*dz) + origin;
					std::vector<FParticle *> neighbors;
					GetNeigboringParticles_cell(i, j, k, -1, 1, -1, 0, -1, 1, neighbors);

					float sum_weight = 0.0;
					float sum_u = 0.0;
					for (FParticle* p : neighbors)
					{
						float weight = 4.0 / 3.0 * M_PI * rho * p_radii * p_radii * p_radii * linear_kernel(p->m_curPosition - pos, dy);
						sum_u += weight * p->m_velocity[1]; //+ p->m_c.col(1).dot(pos - p->m_curPosition);
						sum_weight += weight;
					}

					if (sum_weight != 0.0)
						v.set(i, j, k, sum_u / sum_weight);
					else
						v.set(i, j, k, 0.0);
				}

		// w-component of velocity
#pragma omp for schedule(static)
		for (int k = 0; k < nk + 1; k++)
			for (int j = 0; j < nj; j++)
				for (int i = 0; i < ni; i++)
				{
					Vector3f pos = Vector3f((i + 0.5)*dx, (j + 0.5)*dy, k * dz) + origin;
					std::vector<FParticle *> neighbors;
					GetNeigboringParticles_cell(i, j, k, -1, 1, -1, 1, -1, 0, neighbors);

					float sum_weight = 0.0;
					float sum_u = 0.0;
					for (FParticle* p : neighbors)
					{
						float weight = 4.0 / 3.0 * M_PI * rho * p_radii * p_radii * p_radii * linear_kernel(p->m_curPosition - pos, dz);
						sum_u += weight * p->m_velocity[2]; // +p->m_c.col(2).dot(pos - p->m_curPosition);
						sum_weight += weight;
					}

					if (sum_weight != 0.0)
						w.set(i, j, k, sum_u / sum_weight);
					else
						w.set(i, j, k, 0.0);
				}

		// m-component of velocity
#pragma omp for schedule(static)
		for (int k = 0; k < nk; k++)
			for (int j = 0; j < nj; j++)
				for (int i = 0; i < ni; i++)
				{
					Vector3f pos = Vector3f(i*dx, j*dy, k*dz) + origin;
					std::vector<FParticle *> neighbors;
					GetNeigboringParticles_cell(i, j, k, -1, 1, -1, 1, -1, 1, neighbors);

					float sum_weight = 0.0;
					float sum_u = 0.0;
					for (FParticle* p : neighbors)
					{
						float weight = 4.0 / 3.0 * M_PI * rho * p_radii * p_radii * p_radii * linear_kernel(p->m_curPosition - pos, dz);
						sum_u += weight * p->m_mass; // +p->m_c.col(2).dot(pos - p->m_curPosition);
						sum_weight += weight;
					}

					if (sum_weight != 0.0)
						m.set(i, j, k, sum_u / sum_weight);
					else
						m.set(i, j, k, 0.0);
				}

		// Descriptor creation
#pragma omp for schedule(static)
		for (int n = 0; n < cells_pos.size(); n++)
		{
			Vector3f& pos = cells_pos[n];
			float mass = GetMass(pos);
			Vector3f vel = GetVelocity(pos);
			int idx = 4 * n;
			result[idx + 0] = mass;
			result[idx + 1] = vel[0];
			result[idx + 2] = vel[1];
			result[idx + 3] = vel[2];
		}
	}
}

void LAPIC::GetNeigboringParticles_cell(int i, int j, int k, int wl, int wh, int hl, int hh, int dl, int dh, std::vector<FParticle *>& res)
{
	for (int sk = k + dl; sk <= k + dh; sk++)
		for (int sj = j + hl; sj <= j + hh; sj++)
			for (int si = i + wl; si <= i + wh; si++)
			{
				if (si < 0 || si > ni - 1 || sj < 0 || sj > nj - 1 || sk < 0 || sk > nk - 1)
					continue;
				res.insert(res.end(), cells[(sk * nj * ni) + (sj * ni) + si].begin(), cells[(sk * nj * ni) + (sj * ni) + si].end());
			}
}

Vector3f LAPIC::GetVelocity(Vector3f& pos)
{
	//Interpolate the velocity from the u and v grids
	Vector3f dist = (pos - origin);
	Vector3f p(dist[0] / dx, dist[1] / dy, dist[2] / dz);
	Vector3f p0 = p - Vector3f(0, 0.5, 0.5);
	Vector3f p1 = p - Vector3f(0.5, 0, 0.5);
	Vector3f p2 = p - Vector3f(0.5, 0.5, 0);
	float u_value = interpolate_value(p0, u);
	float v_value = interpolate_value(p1, v);
	float w_value = interpolate_value(p2, w);

	return Vector3f(u_value, v_value, w_value);
}

float LAPIC::GetMass(Vector3f& pos)
{
	Vector3f dist = (pos - origin);
	Vector3f p(dist[0] / dx, dist[1] / dy, dist[2] / dz);
	float m_value = interpolate_value(p, m);

	return m_value;
}
