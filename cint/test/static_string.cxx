#include <string>
#include <iostream>
using namespace std;

class MyClass {
public:
   static std::string configStr;
};

std::string globalStr = "outside"; //  = std::string();

std::string MyClass::configStr = "trap"; //  = std::string();

int func() { // const std::string s = "" ){
   // cerr << (void*)&(globalStr) << endl;
   globalStr="global"; // s
   //cerr << (void*)&(MyClass::configStr) << endl;
   MyClass::configStr="test"; // s
   
   cout << "global: " << globalStr << endl;
   cout << "static: " << MyClass::configStr << endl;
   return ( globalStr=="global" &&  MyClass::configStr=="test");
}

int main() {
   return !func();
}
