#pragma once
#include<iostream>
using namespace std;
class Teacher
{
private:
	int id;
	string name;
public:
	string getName() {
		return name;
	}
	int getid() {
		return id;
	}
	void setName(string nname) {
		name = nname;
	}
	void setid(int iid) {
		id = iid;
	}
	void show() 
	{
		cout << "Teacher's ID is " << id << "\tname is " << name << endl;
	}
	~Teacher() { cout << "Destructor" << endl; };
};

