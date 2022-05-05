#pragma once
#include<iostream>
using namespace std;
class Point
{
    friend class ManagePoint;//友元类
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
    friend double Distance(Point a, Point b);//友员函数，作为全局的友元
    //friend void ManagePoint::showPoint(Point a);
    friend Point operator +(Point a, Point b);//运算友元

    //get和set
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

