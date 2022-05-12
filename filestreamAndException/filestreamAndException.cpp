#include <fstream>
#include <iostream>
#include <string>
using namespace std;

int getInt() {
	string input = "";
	getline(cin, input);
	//判断input里是否包含非数字的char
	//如果出现12e或者e12就不转换了
	bool flag = 1;
	for (char i : input) {
		if (isdigit(i)) continue;
		flag = 0;
		break;
	}
	int num = -1;
	try {
		if (flag) num = stoi(input, 0, 10);
	}
	catch (invalid_argument &e) {							//logic_error
		cout << e.what() << endl;			//exception
		return num;
	}
	return num;
}

int main()
{
    //char line[80];
    //string str;
    //ifstream fhandle("data.txt"); //"data.txt.txt"
    //if (!fhandle.is_open()) cout << "Fail to open file" << endl;
    //while (!fhandle.eof()) { // fhandle.eof() == 0
    //    fhandle.getline(line, 80);  // cin.getline(line,80)
    //    //getline(fhandle,str);     // fhandle.getline(line,'\n')
    //    cout << line << endl;       
    //}

    //fhandle.close();

    //ofstream fhandle2("data.txt");
    //fhandle2 << "Hello world!\nThis is the second line!\nThird line" << endl;

	/*for (int i = 0; i < 3; i++) {
		cout << "Input an integer:\n";
		int j = getInt();
		cout << "You input: " << j << endl;
	}*/

	//string str = "abc";
	//try {
	//	char c = str.at(2);
	//	string t(str.max_size()+1, c);
	//	cout << t << endl;
	//}
	//catch (invalid_argument &e) {
	//	cout << e.what() << endl;
	//}
	//catch (bad_alloc& e) {
	//	cout << e.what() << endl;
	//}
	//catch (length_error &e) {
	//	cout << e.what() << endl;
	//}
	////cout << t << endl;
}
