/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <iostream>
using namespace std;

class A
{
public:
    A &operator() ()
       {cout<<"0 argument"<<endl; return *this;}
    A &operator() (double)
       {cout<<"1 argument"<<endl; return *this;}
    A &operator() (double,double)
       {cout<<"2 argument"<<endl; return *this;}
};

int main()
{
    A a;
    cout<<"---- trial 1: "<<endl;
    a(3.1)(3.1,3.2)(3.2);
    cout<<"---- trial 2: "<<endl;
    a(1.1)(1.1,2.2);
    cout<<"---- trial 3: "<<endl;
    a(1.1)(1.1,2.2)()(1.1);
    cout<<"---- trial 4: "<<endl;
    a()(1.1)(1.1,2.2);
    cout<<"---- trial 5: "<<endl;
    a()(1.1)(1.1,2.2)();
    return 0;
}


