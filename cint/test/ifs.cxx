/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#include <fstream.h>
#else
#include <iostream>
#include <fstream>
using namespace std;
#endif
#include <stdio.h>

// ???
//   VC++6.0: fin.fail() and fin.flags() returns different value 
//   with and without 
// ???

int main()
{
  float a,b;
  int i=0;
  ifstream fin("ifs.data");
  ifstream *pfin;
  pfin = &fin;
  printf("%d %x %x %x\n",fin.eof(),fin.bad(),fin.good(),fin.rdstate());
  while((fin >> a >> b) && i<10) {
    //cout << "a=" << a << " b=" << b << endl;
    printf("a=%g b=%g\n",a,b);
    printf("%d %x %x %x\n",fin.eof(),fin.bad(),fin.good(),fin.rdstate());
    i++;
  }
  printf("%d %x %x %x\n",fin.eof(),fin.bad(),fin.good(),fin.rdstate());
  return 0;
}

