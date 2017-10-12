#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <windows.h>
#include <math.h>

#define PI			3.141592653589793238
#define RADIAN(x)	((x)*PI/180.0)
#define CAMERA_DEPTH	2000.0f

class Camera  
{
public:
	typedef enum {
		VIEW_TOP = 0,
		VIEW_BOTTOM,
		VIEW_FRONT,
		VIEW_BACK,
		VIEW_LEFT,
		VIEW_RIGHT,
		VIEW_PERS,
		VIEW_ORTHO
	} ViewType;
	Camera();
	virtual ~Camera();
	float getOrbit();
	float getLevel();
	void CalculateCameraMatrix(float m[]);
	void CalculateLookAt(float vec[]);
	void CalculateRightUp(float rightVec[], float upVec[]);
	void CalculateDistAngFromRaw();
	void Init(ViewType type, float dis);		// Camera �ʱ�ȭ
	void EditDistance(float target);				// Camera �� Distance�� ����
	void EditPan(float X, float Y);	// Camera �� Panning�� ���� ����
	void EditRotate(float orb, float lat, float lev);	// Camera �� Rotation�� ���� ����
	void EditOnFloor(float uptoFloor, float distEdit);	// �������� y�� ����

	void GetRay(int width, int height, int x, int y, float *startpos, float *direction);
	void SetCameraPos(float x, float y, float z)	{posX = x;	posY = y;	posZ = z;}
	void SetTargetPos(float x, float y, float z)	{tarX = x;	tarY = y;	tarZ = z;}
	void SetUpVect(float x, float y, float z)		{norX = x;	norY = y;	norZ = z;}

//private:
	ViewType	type;				// Camera Ÿ��
	float		posX, posY, posZ;	// Camera �� ��ġ
	float		tarX, tarY, tarZ;	// Camera �� ��ǥ��
	float		norX, norY, norZ;	// Camera �� ��������
	float		distance;			// Camera �� ��ǥ������ �Ÿ�
	float		angOrb, angLat, angLev;	// Camera Angle
	float		fov;				// ȭ��
	float		floorBlend;		// floor �������� �ø���� �ٷ� �װ�
	int			floorConstraint;	// floor Constraint
};

#endif // __CAMERA_H__