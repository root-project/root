/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

#include <vector>
using namespace std;

typedef vector<double> vd;

int main() {
  vector<double> *v1 = new vector<double>[3];
  delete[] v1;
  vd *v3 = new vd[3];
  delete[] v3;
  vector<double> *v5 = new vd[3];
  delete[] v5;
  vd *v7 = new vector<double>[3];
  delete[] v7;

  vector<double> *v2 = new vector<double>;
  delete v2;
  vd *v4 = new vd;
  delete v4;
  vector<double> *v6 = new vd;
  delete v6;
  vd *v8 = new vector<double>;
  delete v8;

  printf("success\n");
  
  return 0;
}
