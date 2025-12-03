#ifndef MYCLASS_H
#define MYCLASS_H

#include <map>
#include <string>

using namespace std;

class MyClass{
public:
    MyClass(){}
    virtual ~MyClass(){}
    map<string,double>& Param() {return fParam;}
private:
    map<string,double> fParam;
};


#if defined(__MAKECINT__)
#pragma link C++ class MyClass;
#endif

#endif
