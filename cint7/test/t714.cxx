/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <stdio.h>

#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

void f1(float eta=.9) {
  bool xx;
  if ( 1 && ( (eta > 0.98 && eta < 1.02) ||
	      (eta > 1.38 && eta < 1.42) ||
	      (eta > 0.88 && eta < 0.92) )) {
    printf("TRUE\n");
    xx=true;
  }
  else {
    printf("FALSE\n");
    xx=false;
  }
  
}

void f2(float eta=.9) {
  bool xx;
  if (1 && ( eta > 0.98 && eta < 1.02 ||
	     eta > 1.38 && eta < 1.42 ||
	     eta > 0.88 && eta < 0.92 ) ) { 
    printf("TRUE\n");
    xx=true;
  }
  else {
    printf("FALSE\n");
    xx=false;
  }
}

void yy ()
{
  for (int iev = 0; iev <2; iev++) {
    f2(0.9); // t
    f1(0.9); // t
    f2(1.4); // t
    f1(1.4); // t
    f2(1.0); // t
    f1(1.0); // t
    f2(0.8); // f
    f1(0.8); // f
    f2(1.5); // f
    f1(1.5); // f
    f2(0.95); // f
    f1(0.95); // f
    f2(1.1); // f
    f1(1.1); // f
  }
}


#include <iostream>
void RunGMT1_cutJul2001()
{
  float eta = .9;
  bool xx;
  if ( 1 && ( (eta > 0.98 && eta < 1.02) ||
	      (eta > 1.38 && eta < 1.42) ||
	      (eta > 0.88 && eta < 0.92) )) {
    cout << "TRUE w. brackets   " ;
    xx=true;
  }
  else {
    cout << "FALSE w. brackets  " ;
    xx=false;
  }

  if (1 && ( eta > 0.98 && eta < 1.02 ||
	     eta > 1.38 && eta < 1.42 ||
	     eta > 0.88 && eta < 0.92 ) ) { 
    cout << "true without " << endl;
  }
  else
    {
      cout << "false without" << endl;
      if (xx) cout << "*** oops !!! *** eta = " << eta << endl;

    }
}

void xx ()
{
  RunGMT1_cutJul2001();
  for (int iev = 0; iev <2; iev++) {
    RunGMT1_cutJul2001();
  }
}

int main() {
  xx();
  yy();
  return 0;
}
