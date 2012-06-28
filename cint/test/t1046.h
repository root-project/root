/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// demo/exception/ehdemo.h
// This source has to be compiled. Refer to setup

#include <exception>
#include <string>
#include <stdio.h>
#ifndef __hpux
using namespace std;
#endif


class IntVar {
  int *d;
  int r,c;
 public:
  IntVar(int iRows, int iCols) : r(0), c(0) {
    if (iRows*iCols > 100) {
      throw exception();
      //throw exception("IntVar too big");
    }

    d = new int[iRows*iCols];
    r=iRows;
    c=iCols;
  }
  ~IntVar() { 
    if(r&&c) delete[] d; 
  }
  void disp() const { printf("r=%d c=%d\n",r,c); }

};

