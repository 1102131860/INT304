#pragma once
#include"Point.h";

class ThreeDPoint : public Point
{
private:
	double z;
	using Point::setX;
	using Point::setY;
public:
	ThreeDPoint(double x, double y, double z = 0) : Point(x,y) {
		this->z = z;
	};
	~ThreeDPoint() {
		cout << "ThreeDpoint destructor\n";
	}
	
};


