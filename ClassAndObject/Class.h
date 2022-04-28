#pragma once
#include<vector>
#include"Student.h";
#include"Teacher.h";
class Class
{
private:
	Teacher t;
	Student st;
public:
	Class(string sname, int sage) : st(sname, sage) 
	{
		string tname;
		int tid;
		cout << "Input the teacher's name: ";
		cin >> tname;
		t.setName(tname);
		cout << "Input the teacher's id: ";
		cin >> tid;
		t.setid(tid);
	}
	void show()
	{
		t.show();
		st.show();
	}
};

