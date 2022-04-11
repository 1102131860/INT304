#include<iostream>;
#include<string>;
#include <vector>;
using namespace std;
enum Ball
{
    red,
    yellow,
    blue
};
struct Student
{
    string name;
    int age;
    void setname(string newname) {
        name = newname;
    }
};
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (auto& el : vec)
    {
        os << el << ' ';
    }
    return os;
}

int main()
{
    {//sizeof() and pointer
        /*double da[5];
        int db[] = { 1,2,3,4,5 };
        int* p;
        double* q;
        cout << sizeof(da) << endl;
        cout << sizeof(da[1]) << endl;
        cout << sizeof(db) << endl;
        cout << sizeof(db[1]) << endl;
        cout << sizeof(p) << endl;
        cout << sizeof(q) << endl;
        cout << &da << endl;
        cout << &da[0] << endl;
        cout << &da[1] << endl;
        cout << &da[4] << endl;
        p = &db[1];
        cout << *p << endl;
        cout << &db[1] << endl;
        cout << p << endl;
        cout << &p << endl;*/
    }
    
    {//char array VS string
        //char greeting[5] = { 'H','e','l','l','o' };         // char array can be altered
        //char Greeting[6] = { 'H','e','l','l','o','\0' };
        //cout << greeting << endl << Greeting << endl;
        //Greeting[1] = 'E';
        //cout << Greeting << endl;

        //string name = "Elon";      // in C++, string can also be altered
        //cout << name << endl;
        //name = name + " Reeve "; // concatenates two strings
        //string famname = "Musk";
        //name += famname;
        //cout << name << endl;
        //name[0] = 'e';          // not "e", char NOT string
        //cout << name << endl;

     // string member methods
        //char& address = name.front();
        //address = 'E';
        //cout << name << endl;
        //cout << address << "\t" << &address << endl;    // string也是一种array, 只要是array它的地址和里面首个item的地址是一样的
        //char& lastletter = name.back();
        //cout << lastletter << "\t" << &lastletter << endl;
        //name.pop_back();
        //cout << name << endl;
        //name.push_back('k');
        //cout << name << endl;

        //name.erase(4, 5);   // 从第index4开始擦除，擦除5个chars
        //cout << name << endl;
        //如果去除所有‘e’改怎么做？
        //for (string::iterator i = name.begin(); i != name.end(); i++) {     
        //    if (*i == 'e') {
        //        name.erase(i);
        //        i--;
        //    }
        //}
        //cout << name << endl;
        //name.insert(6,2,'e'); // 在index6前面加俩个‘e'
        //cout << name << endl;
        //找出不重复的letter
        //for (string::iterator i = name.begin(); i != name.end();i++) {
        //    if (name.find(*i) == name.rfind(*i)) cout << *i << "\t";
        //}
        //cout << "\n" << name.substr() << endl; //取出所有
        //cout << name.substr(5, 6) << endl;  //从index5后面取6个chars
        

    // string non-member methods
        /*string number = "10100111";
        int integer = stoi(number, 0, 2);
        cout << integer << endl;
        double doub = stod(number, 0);
        cout << doub << endl;
        string sum = to_string(123) + "1";
        cout << sum << endl;*/
    }
    
    {//Pointer & array
        //int intArray[3] = { 1,4,10 };
        //int* ptr = intArray; // int* ptr = &intArray[0];
        //*(ptr + 1) = *(ptr + 1) / 2;
        //cout << intArray[1] << endl;

        //cout << "Input the length of int[]: ";
        //int num;
        //cin >> num;
        ////    int intArray[num];
        //int* ptr = new int[num];
        //for (int i = 0; i < num; i++) {
        //    cout << "Input Value " << i + 1 << ": ";
        //    cin >> *(ptr + i);
        //}
        //for (int* i = ptr; i != ptr + num; i++) {
        //    cout << *i << "\t";
        //}
    }

    {//enumertaion & structure
        ////Ball b1 = Ball(0); 
        //Ball b1 = (Ball)0;
        //if (b1 == 0) cout << "\n" << "Ball b1 is red" << endl;
        //else cout << "Ball b1 is not red" << endl;

        //Student s = { "John",25 };
        //Student s1;
        //s1.name = "Jenny";
        //s1.age = 23;

        //cout << "student s name :\t" << s.name << "\tage: " << s.age << endl;
        //Student* sptr = &s1;
        //cout << "student s1 name :\t" << sptr->name << "\tage: " << sptr->age << endl;
    
        //s.setname("George");
        //cout << "student s new name : " << s.name << endl;
    }

    {//vector VS array
        //cout << "Input the number of integer you want to insert into vector : ";
        //int num;
        //cin >> num;
        //vector<int> intVector(num);
        //for (int i = 0; i < num; i++) {
        //    cout << "Input integer " << i + 1 << ": ";
        //    cin >> intVector[i];
        //}
        //for (int e : intVector) {
        //    cout << e << "\t";
        //}

     //vector member methods
        //vector<int> a{ 1,2,3 }, b{ 7,5 };
        //a.push_back(4);
        //cout << a << endl;
        //int n = b.back();
        //b.pop_back();
        //cout << b << endl;
        //a.erase(a.begin() + 1, a.begin() + 2); // index1被擦掉，[a,b), paramter必须是iterator
        //cout << a << endl;
        //a.insert(a.begin(),2);  //在index0之前插入2
        //cout << a << endl;
        //a.insert(a.end(), 3, 8);    //在最后插入3个8
        //cout << a << endl;
        //a.insert(a.end(), b.begin(), b.end());  // 相当于append，concatenate,连接两个vector
        //cout << a << endl;
        ////insert 之前; substr和erase 之后[) 
    }
}


