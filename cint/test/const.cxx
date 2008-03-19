/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream>
using namespace std;

//#define max 5
//const int max=5;
void test1() {
  static const int max = 5;
  //const int max = 5;
  int b[max];
  int i=0;
  b[i]=0;
  for(i=0;i<max;i++) {
    b[i]=i;
    cout << b[i] << endl;
  }

}

int main() {
  
  for(int i=0;i<3;i++) test1();

  return 0;
}
