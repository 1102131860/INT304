#include"ThreeDpoint.h";
#include"Student.h";
double Distance(Point a, Point b) {
    double x = a.x - b.x;
    double y = a.y - b.y;
    return sqrt(x * x + y * y);
}

Point operator + (Point a, Point b) {
    Point newpoint(a.x + b.x, a.y + b.y);
    return newpoint;
}
int main()
{
   Point p1(1, 2), p2 (2,-3);
    ManagePoint mp1;
    mp1.showPoint(p1);
    cout << "Destructor\n";
    mp1.showPoint(p1 + p2);

    ThreeDPoint Threep3(1, 3, 5);
    cout << "Destructor\n";

    /*Student s1;
    s1.show();
    Postgraduate pg1;
    pg1.show();
    pg1.Student::show();
    pg1.show(pg1);*/
    
    /*Student* s2 = new Postgraduate;
    s2->show();
    delete s2;*/

    /*Student *s;
    Postgraduate pg;
    PhDStudent phds;
    cout << "1: Postgraduate\telse: PhD student\n";
    int choice = 1;
    cin >> choice;
    if (choice == 1)
        s = &pg;
    else
        s = &phds;
    s->show();*/


}


