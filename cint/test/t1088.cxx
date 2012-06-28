/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <string>
#include <iostream>

using namespace std;

namespace test {
  class A {
  public:
    string i;
    operator string() const {return i;}
  };
  
  bool operator== (const A& a1, const A& a2) {
    cout<<"operator== (const A&, const A&) called"<<endl;
    return a1.i == a2.i;
  }
}

using namespace test;

int main() {
  A a1,a2;
  a1.i = "something";
  a2.i = "something";
  
  //a1 == a2;
  cout<<(a1 == a2)<<endl;
  return 0;
}







