/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

class ehx {
};

class eh1 {
  int err;
 public:
  eh1(int errin=123) { err=errin; cout << "er1(" << err << ")" << endl; }
#ifdef DTOR
  ~eh1() { cout << "~er1(" << err << ")" << endl; }
#endif
  void disp() { cout << "err=" << err << endl; }
};

void test1()
{
  try {
    cout << "eh test1------------------" << endl;
    throw eh1();
    cout << "never comes here" << endl;
  }
  catch(ehx& x) {
    cout<< "ehx" << endl;
  }
  catch(eh1& x) {
    cout<< "eh1" << endl;
    x.disp();
  }
  catch (...) {
    cout<< "eh default" << endl;
  }
}

void test2()
{
  try {
    cout << "eh test2------------------" << endl;
    throw eh1();
    cout << "never comes here" << endl;
  }
  catch(ehx& x) {
    cout<< "ehx" << endl;
  }
  catch (...) {
    cout<< "eh default" << endl;
  }
}

void sub3(int e) {
  cout << "sub3()" << endl;
  throw eh1(e);
  cout << "never sub3" << endl;
}

void test3()
{
  try {
    cout << "eh test3------------------" << endl;
    sub3(3);
    cout << "never comes here" << endl;
  }
  catch(ehx& x) {
    cout<< "ehx" << endl;
  }
  catch(eh1& x) {
    cout<< "eh1" << endl;
    x.disp();
  }
  catch (...) {
    cout<< "eh default" << endl;
  }
}

void test4()
{
  try {
    cout << "eh test4------------------" << endl;
    sub3(4);
    cout << "never comes here" << endl;
  }
  catch(ehx& x) {
    cout<< "ehx" << endl;
  }
  catch (...) {
    cout<< "eh default" << endl;
  }
}


int main() 
{
  test1();
  test2();
  test3();
  test4();
  return 0;
}
