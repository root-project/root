/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/********************************
 * test.h
 *
 ********************************/

#include <iostream>

class X
{
 public:
  static const int i;
  static const int a[];

  void print();
};

const int X::a[]={0,1,2,3,4,5,6,7,8,9};

void X::print()
{
    std::cout << "Print static const int member:\n"
              << X::a[0] << '\t'
              << X::a[1] << '\t'
              << std::endl;
    for(int i=0;i<10;i++) std::cout << X::a[i] << " ";
    std::cout << std::endl;
}

/*****************************************
 * test.C
 *
 *****************************************/
#include <iostream>
//#include "test.h"

using namespace std;

int main()
{
  cout << "Hello World!\n";

  X x;
  x.print();
}

