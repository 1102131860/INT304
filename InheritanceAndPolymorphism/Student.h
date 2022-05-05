#pragma once
#include<iostream>
using namespace std;
class Student
{
public:
	virtual void show();
	~Student() { cout << "Destruct student\n"; }
};

class Postgraduate : public Student
{
public:
	void show();
	void show(Postgraduate ps);
};

class PhDStudent : public Student
{
public:
	void show();
};