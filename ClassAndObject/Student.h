#pragma once
#include<iostream>
using namespace std;
class Student
{
private:
	string name = "";
	int age = 0;
public:
	/*Student();*/
	Student(string name,  int age) {
		this->name = name;
		this->age = age;
	}
	void show();
	Student(Student& c);
};
//Student::Student() {
//	name = "";
//	age = 0;
//}
