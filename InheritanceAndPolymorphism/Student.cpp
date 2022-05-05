#include "Student.h"

void Student::show() {
	cout << "This is student's show function!\n";
}

void Postgraduate::show() {
	cout << "This is postgraduate's show function\n";
}

void Postgraduate::show(Postgraduate ps) {
	cout << "This is postgraduate's show function!¡ª¡ª overloading\n";
}

void PhDStudent::show() {
	cout << "This is PhD student's show function!\n";
}