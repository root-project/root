/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <iostream>
using namespace std;

int main() {
  long double x=0.12;
  long long y= 34;
  unsigned long long z= 56;
  long double result;
  result = x + y;
  //result = y ;
  //cout << x << " " << y << endl;
  cout << result << endl;
  result += z;
  cout << result << endl;
  result += y;
  cout << result << endl;
  z += y;
  cout << z << endl;
  y += z;
  cout << y << endl;
  return(0);
}
