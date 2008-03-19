/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#if defined(__hpux) || defined(__BCPLUSPLUS__) 

#define STDIN "stdio.h"
#define IOSTREAM "iostream.h"
#include STDIN
#include IOSTREAM

#else

#define STDIN "stdio.h"
#define IOSTREAM "iostream"
#include STDIN
#include IOSTREAM
using namespace std;

#endif

int main() {
  char buf[100];
  sprintf(buf,"Hello world with printf\n");
  cout << buf ;
  cout << "Hello world with iostream" << endl;
  return 0;
}


