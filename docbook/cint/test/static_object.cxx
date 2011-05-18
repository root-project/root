#include "static_object.h"

class MyClass {
public:
   static Object configStr;
};

Object globalStr = 2; 

static Object staticGlobalStr; //  = 3;

Object MyClass::configStr = 4; 

int func() { 
   //cerr << "global:" << (void*)&(globalStr) << endl;
   // globalStr="global"; // s
   //cerr << "class: " << (void*)&(MyClass::configStr) << endl;
   //MyClass::configStr="test"; // s
   cout << "global: " << globalStr.value << endl;
   cout << "class : " << MyClass::configStr.value << endl;
   return ( globalStr.value == 2 &&  MyClass::configStr.value == 4 );
}

int main() {
   return !func();
}
