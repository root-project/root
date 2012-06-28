/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// ---- proba.C
#include <iostream>
using namespace std;

class A
{
public:
    static A& create(int i);
    A& operator() (int i);
    A& operator() (int i,int j);
    A& operator[] (int i);
};

A &A::create(int i)
{
    A *a = new A;
    cout<<"A::create("<<i<<")"<<endl;
    return *a;
}

A &A::operator()(int i)
{
    cout<<"A::operator()("<<i<<")"<<endl;
    return *this;
}

A &A::operator()(int i,int j)
{
  cout<<"A::operator()("<<i<<","<<j<<")"<<endl;
  return *this;
}

A &A::operator[](int i)
{
    cout<<"A::operator[]("<<i<<")"<<endl;
    return *this;
}

int main()
{
  A* p1 = &A::create(1)(2);
  A* p2 = &A::create(1)(2,3);
  A* p3 = &A::create(1)[2];
  delete p3;
  delete p2;
  delete p1;
  return 0;
}




