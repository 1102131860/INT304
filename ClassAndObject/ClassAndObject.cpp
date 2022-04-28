#include<string>
#include<iostream>
#include<vector>
#include"Student.h";
#include"Teacher.h";
#include"Class.h";
using namespace std;

int main()
{
    //vector实现储存和展示
    {
        /*
        cout << "Input the number of students" << endl;
        int num;
        cin >> num;

        vector<int> ID(num);
        vector<int> age(num);
        vector<string> name(num);
        vector<string> major(num);

        //数据录入
        int iid, iage;
        string iname, imajor;
        for (int i = 0; i < num; i++) {
            cout << "Input student " << i + 1 << " ID: ";
            cin >> iid;
            ID[i] = iid;
            cout << "Input student " << i + 1 << " name: ";
            cin >> iname;
            name[i] = iname;
            cout << "Input student " << i + 1 << " age: ";
            cin >> iage;
            age[i] = iage;
            cout << "Input student " << i + 1 << " major: ";
            cin >> imajor;
            major[i] = imajor;
        }
        //数据输出
        for (int i = 0; i < num; i++) {
            cout << "Student " << i + 1 << "\tid " << age[i]
                << "\tname " << name[i] << "\tage " <<
                age[i] << "\tmajor " << major[i] << endl;
        }
        */
    }
    
    //structure
    {//注意括号
        /*
        struct Student      
        {
            int id;
            string name;
            int age;
            string major;
        };
        cout << "Input the number of students" << endl;
        int num;
        cin >> num;
        Student* ptr = new Student[num];

        int iid, iage;
        string iname, imajor;
        for (int i = 0; i < num; i++) {
            Student s;
            cout << "Input student " << i + 1 << " ID: ";
            cin >> iid;
            s.id = iid;
            cout << "Input student " << i + 1 << " name: ";
            cin >> iname;
            s.name = iname;
            cout << "Input student " << i + 1 << " age: ";
            cin >> iage;
            s.age = iage;
            cout << "Input student " << i + 1 << " major: ";
            cin >> imajor;
            s.major = imajor;
            *(ptr + i) = s;
        }

        for (int i = 0; i < num; i++) {
            cout << "Student " << i + 1 << "\tid " << (ptr + i)->id
                << "\tname " << (ptr + i)->name << "\tage " << (ptr + i)->age
                << "\tmajior " << (ptr + i)->major << endl;
        }
        */
    }
    
    //class
    {//注意括号的范围
        //class Student 
        //{
        //public:
        //    int id, age;
        //    string name, major;
        //    //构造器
        //    /*Student(int id = 0, int age = 0, string name = "",string major = "") {
        //        this->id = id;
        //        this->age = age;
        //        this->name = name;
        //        this->major = major;
        //    }*/
        //    void showself() {
        //        cout << "Student id " << id << "\tname " << name
        //            << "\tage " << age << "\tmajor " << major << endl;
        //    }
        //};
        //cout << "Input the number of students" << endl;
        //int num;
        //cin >> num;
        //Student* ptr = new Student[num];

        //int iid, iage;
        //string iname, imajor;
        //for (int i = 0; i < num; i++) {
        //    Student s;
        //    cout << "Input student " << i + 1 << " ID: ";
        //    cin >> iid;
        //    s.id = iid;
        //    cout << "Input student " << i + 1 << " name: ";
        //    cin >> iname;
        //    s.name = iname;
        //    cout << "Input student " << i + 1 << " age: ";
        //    cin >> iage;
        //    s.age = iage;
        //    cout << "Input student " << i + 1 << " major: ";
        //    cin >> imajor;
        //    s.major = imajor;
        //    *(ptr + i) = s;
        //}
        //for (int i = 0; i < num; i++) {
        //    (ptr + i)->showself();
        //}
    }

    //constructor
    //调用头文件里的class student
    {
        //Student s("John",12);

    }
    
    //access specifer
    //Destructor
    {
       /* Teacher t;
        t.setid(1);
        t.setName("Miller");
        cout << "Teacher's name is " << t.getName() << " id is " << t.getid() << endl;*/
    }

    //Class Composition
    {
        Class c("George",22);
        c.show();
    }
}

