#include <stdio.h>
#include <iostream>
#include <algorithm>
using namespace std;
double larger(double a, double b);
void showSum(double a, int b) {
    cout << "the sum of " << a << " and " << b << " is " << a + b;
}
int pointerSum(int* a, int* b) {
    for (int i = 0; i < *b; i++) {
        *a += *a;
    }
    return *a;
}
int copySum(int a, int b) {
    for (int i = 0; i < b; i++) {
        a += a;
    }
    return a;
}
double Sum(double a, int b) {
    return a + b;
}
double Sum(int a, double b) {
    return a + b;
}
class ComplexNum {
    double x, y;
public:
    ComplexNum(int x = 0, int y = 0) {  // 函数的默认参数(default paramter)
        this->x = x;                    // 如果没有输入参数，那么默认this->x = 0, this->y = 0
        this->y = y;
    }
    ComplexNum operator + (ComplexNum &z) {
        ComplexNum z1(x + z.x, y + z.y);
        return z1;
    }
    void show() {
        cout << "real part: " << x << " imaginary part: " << y << endl;
    }
};
class Student{
    string name;
    int age;
 public:
     Student(string name = "", int age = 0) {
         this->name = name;
         this->age = age;
     }
    bool operator < (Student s) {
         return age < s.age ? 1 : 0;
    }
    string getName() {
        return name;
    }
};
int* input() {
    cout << "Please input the total number of integers: ";
    int num;
    cin >> num;
    int* ptr = new int[num];
    *ptr = num;
    for (int i = 0; i < num; i++) {
        cout << "\nInput integer " << i + 1 << " : ";
        cin >> *(ptr + i + 1);
    }
    return ptr;
}


int main()
{
    //如何使用函数
    {
        /*double x = sqrt(128), y = exp(2.5);
        cout << "x: " << x << " y: " << y << endl;
        cout << "The larger one is " << larger(x, y) << endl;*/
    }
    //参数时指针和变量本身的区别
    {
        //int c = 3, d = 2;
        ///*cout << pointerSum(&c, &d) << endl;
        //cout << c << endl;*/
        //cout << copySum(c, d) << endl;
        //cout << c << endl;
    }
    //Overloading
    {
        //cout << Sum(3.1, 5) << endl;
    }
    //Operator Overloading
    {
        //ComplexNum z1, z2(1,2);
        //z1.show();              //default paramter
        //ComplexNum z3(2,5);
        //z1 = z2 + z3;           
        //z1.show();

        //Student s1("John", 20), s2("George", 21);
        //cout << s1.getName() << " is youngeer than " << s2.getName() << " is " << (s1 < s2);
    }
    //返回多值
    {
        //请用户输入想要输入的整数个数，然后让用户输入那些数，最后返回一个数组
        //int *ptr = input();
        //for (int i = 0; i < *ptr; i++) {
        //    cout << *(ptr + i + 1) << "\t";
        //}
        ////sort（排序）输入的数
        ////先声明#include <algorithm>
        //sort((ptr + 1), (ptr + 1 + *ptr),greater<int>()); //参数时起始迭代器和末尾迭代器，sort（it,it+n,cmp) less<int>()
        //cout << endl;
        //for (int i = 0; i < *ptr; i++) {
        //    cout << *(ptr + i + 1) << "\t";
        //}
    }
}

double larger(double a, double b)
{
    return a <= b ? b : a ;
}
