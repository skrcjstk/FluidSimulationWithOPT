#include "APICSim.h"


void APICSim::Initialize(Vector3f p_origin, Vector3f p_bSize, Vector3i p_nCount, float p_rho)
{
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

	cells.resize(nk*nj*ni);
	cells_pos.resize(nk*nj*ni);
	for (int k = 0; k < nk; k++)
		for (int j = 0; j < nj; j++)
			for (int i = 0; i < ni + 1; i++)
			{
				cells_pos[(k*nj*ni) + j*ni + i] = Vector3f(i * dx, (j)*dy, (k)*dz) + origin;
			}


	printf("APIC Grid(%d, %d, %d)\n", ni, nj, nk);
	printf("APIC dx(%.2f) dy(%.2f) dz(%.2f)\n", dx, dy, dz);
}

void APICSim::AssignCells(FluidWorld* p_world)
{
	for (int k = 0; k < nk; k++)
		for (int j = 0; j < nj; ++j)
			for (int i = 0; i < ni; ++i) 
			{
				cells[(k*nj*ni) + j*ni + i].clear();
			}	

	// Store Into The Cells
	int np = p_world->GetNumOfParticles();

	for (int n = 0; n<np; n++) 
	{
		FParticle *p = p_world->GetParticle(n);

		int pi = (int)((p->m_curPosition[0] - origin[0]) / dx);
		int pj = (int)((p->m_curPosition[1] - origin[1]) / dy);
		int pk = (int)((p->m_curPosition[2] - origin[2]) / dz);
		int i = max(0, min(ni - 1, pi));
		int j = max(0, min(nj - 1, pj));
		int k = max(0, min(nk - 1, pk));
		cells[(k*nj*ni) + j*ni + i].push_back(p);
	}
}

void APICSim::Map_P2G(FluidWorld* p_world)
{
	float radii = p_world->GetParticleRadius();

	// u-component of velocity
	for (int k = 0; k < nk; k++)
		for (int j = 0; j < nj; j++)
			for (int i = 0; i < ni+1; i++)
			{
				Vector3f pos = Vector3f(i * dx, (j+0.5)*dy, (k+0.5)*dz) + origin;
				std::vector<FParticle *> neighbors;
				GetNeigboringParticles_cell(i, j, k, -1, 0, -1, 1, -1, 1, neighbors);

				float sum_weight = 0.0;
				float sum_u = 0.0;
				for (FParticle* p : neighbors)
				{
					float weight = 4.0 / 3.0 * M_PI * rho * radii * radii * radii * linear_kernel(p->m_curPosition - pos, dx);
					sum_u += weight * (p->m_velocity[0] + p->m_c.col(0).dot(pos - p->m_curPosition));
					sum_weight += weight;
				}

				if (sum_weight != 0.0)
					u.set(i, j, k, sum_u / sum_weight);
				else
					u.set(i, j, k, 0.0);
			}

	// v-component of velocity
	for (int k = 0; k < nk; k++)
		for (int j = 0; j < nj + 1; j++)
			for (int i = 0; i < ni; i++)
			{
				Vector3f pos = Vector3f((i+0.5)*dx, j * dy, (k+0.5)*dz) + origin;
				std::vector<FParticle *> neighbors;
				GetNeigboringParticles_cell(i, j, k, -1, 1, -1, 0, -1, 1, neighbors);

				float sum_weight = 0.0;
				float sum_u = 0.0;
				for (FParticle* p : neighbors)
				{
					float weight = 4.0 / 3.0 * M_PI * rho * radii * radii * radii * linear_kernel(p->m_curPosition - pos, dx);
					sum_u += weight * (p->m_velocity[1] + p->m_c.col(1).dot(pos - p->m_curPosition));
					sum_weight += weight;
				}

				if (sum_weight != 0.0)
					v.set(i, j, k, sum_u / sum_weight);
				else
					v.set(i, j, k, 0.0);
			}
	
	// w-component of velocity
	for (int k = 0; k < nk + 1; k++)
		for (int j = 0; j < nj; j++)
			for (int i = 0; i < ni; i++)
			{
				Vector3f pos = Vector3f((i+0.5)*dx, (j+0.5)*dy, k * dz) + origin;
				std::vector<FParticle *> neighbors;
				GetNeigboringParticles_cell(i, j, k, -1, 1, -1, 1, -1, 0, neighbors);

				float sum_weight = 0.0;
				float sum_u = 0.0;
				for (FParticle* p : neighbors)
				{
					float weight = 4.0 / 3.0 * M_PI * rho * radii * radii * radii * linear_kernel(p->m_curPosition - pos, dx);
					sum_u += weight * (p->m_velocity[2] + p->m_c.col(2).dot(pos - p->m_curPosition));
					sum_weight += weight;
				}

				if (sum_weight != 0.0)
					w.set(i, j, k, sum_u / sum_weight);
				else
					w.set(i, j, k, 0.0);
			}
}
void APICSim::UpdateAffineMatrix(FluidWorld* p_world)
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

void APICSim::GetNeigboringParticles_cell(int i, int j, int k, int wl, int wh, int hl, int hh, int dl, int dh, std::vector<FParticle *>& res)
{
	for (int sk = k + dl; sk <= k + dh; sk++)
		for (int sj = j + hl; sj <= j + hh; sj++)
			for (int si = i + wl; si <= i + wh; si++)
			{
				if (si < 0 || si > ni - 1 || sj < 0 || sj > nj - 1) continue;
				res.insert(res.end(), cells[sj * ni + si].begin(), cells[sj * ni + si].end());
			}
}

Vector3f APICSim::GetVelocity(Vector3f& pos)
{
	//Interpolate the velocity from the u and v grids
	Vector3f dist = (pos - origin);
	Vector3f p(dist[0] / dx, dist[1] / dy, dist[2] / dx);
	Vector3f p0 = p - Vector3f(0, 0.5, 0.5);
	Vector3f p1 = p - Vector3f(0.5, 0, 0.5);
	Vector3f p2 = p - Vector3f(0.5, 0.5, 0);
	float u_value = interpolate_value(p0, u);
	float v_value = interpolate_value(p1, v);
	float w_value = interpolate_value(p2, w);

	return Vector3f(u_value, v_value, w_value);
}

inline float APICSim::interpolate_value(Vector3f& point, APICArray3d::Array3d<float>& grid)
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

inline Vector3f APICSim::affine_interpolate_value(Vector3f& point, APICArray3d::Array3d<float>& grid)
{
	int i, j, k;
	float fx, fy, fz;

	get_barycentric(point[0], i, fx, 0, grid.width);
	get_barycentric(point[1], j, fy, 0, grid.height);
	get_barycentric(point[2], k, fz, 0, grid.depth);

	return grad_trilerp(grid.get(i, j, k), grid.get(i + 1, j, k), grid.get(i, j + 1, k), grid.get(i + 1, j + 1, k),
		grid.get(i, j, k + 1), grid.get(i + 1, j, k + 1), grid.get(i, j + 1, k + 1), grid.get(i + 1, j + 1, k + 1),
		fx, fy, fz);
}