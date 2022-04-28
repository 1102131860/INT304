#include "Student.h"
void Student::show() {
	cout << "Student's name:\t" << name  << "\tage:\t" << age << endl;
}
Student::Student(Student& c) {
	name = c.name;
	age = c.age;
}