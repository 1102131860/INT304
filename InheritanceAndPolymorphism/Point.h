#pragma once
#include<iostream>
using namespace std;
class Point
{
    friend class ManagePoint;//��Ԫ��
private:
    double x, y;
public:
    Point(double x = 0, double y = 0) {
        this->x = x;
        this->y = y;
    }
    ~Point() {
        cout << "Point destructor\n";
    }
    friend double Distance(Point a, Point b);//��Ա��������Ϊȫ�ֵ���Ԫ
    //friend void ManagePoint::showPoint(Point a);
    friend Point operator +(Point a, Point b);//������Ԫ

    //get��set
    double getX() {
        return x;
    }
    double getY() {
        return y;
    }
protected:
    double setX(double x) {
        this->x = x;
    }
    double setY(double y) {
        this->y = y;
    }
};

class ManagePoint
{
public:
    void showPoint(Point a) {
        cout << "x: " << a.x << " y: " << a.y << endl;
    }
};

