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
	void Init(ViewType type, float dis);		// Camera 초기화
	void EditDistance(float target);				// Camera 의 Distance를 조정
	void EditPan(float X, float Y);	// Camera 의 Panning에 따른 조정
	void EditRotate(float orb, float lat, float lev);	// Camera 의 Rotation에 따른 조정
	void EditOnFloor(float uptoFloor, float distEdit);	// 월드축은 y로 고정

	void GetRay(int width, int height, int x, int y, float *startpos, float *direction);
	void SetCameraPos(float x, float y, float z)	{posX = x;	posY = y;	posZ = z;}
	void SetTargetPos(float x, float y, float z)	{tarX = x;	tarY = y;	tarZ = z;}
	void SetUpVect(float x, float y, float z)		{norX = x;	norY = y;	norZ = z;}

//private:
	ViewType	type;				// Camera 타입
	float		posX, posY, posZ;	// Camera 의 위치
	float		tarX, tarY, tarZ;	// Camera 의 목표점
	float		norX, norY, norZ;	// Camera 의 법선벡터
	float		distance;			// Camera 와 목표점간의 거리
	float		angOrb, angLat, angLev;	// Camera Angle
	float		fov;				// 화각
	float		floorBlend;		// floor 위쪽으로 올리라는 바로 그것
	int			floorConstraint;	// floor Constraint
};

#endif // __CAMERA_H__