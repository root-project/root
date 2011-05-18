/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

class TVector {
  int y;
public:
  TVector(int yin) { 
    y=yin; printf("TVector(%d)\n",y); fflush(stdout);
  }
  ~TVector() { 
    printf("~TVector()\n"); fflush(stdout);
  }
  int Get() const { return y; }
};

class TMatrixRow {
  int x;
public:
  TMatrixRow(int xin) { 
    x=xin; printf("TMatrixRow(%d)\n",x); fflush(stdout);
  }
  void operator=(const TVector& vec) { 
    x = vec.Get();
    printf("TMatrixRow::operator=(TVector(%d))\n",x); fflush(stdout);
  }
  ~TMatrixRow() { 
    printf("~TMatrixRow()\n"); fflush(stdout);
  }
};


