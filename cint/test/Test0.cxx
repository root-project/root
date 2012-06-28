/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Test0.cxx

#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif
#include "MyString.h"

int main()
{
  // Generation and initialization of MyString object
  MyString n1 = MyString("Akira"); //init1
  MyString n2("Toshio");           //init2
  MyString f1 = "Sato";            //init3
  MyString f2;                     //init4

  // assignment of string
  f2 = "Sato"; // from char*
  f2 = f1;     // from MyString

#if 0
  // display contents of MyString
  cout << "n1=" << n1 << "  f1=" << f1 << endl;
  cout << "n2=" << n2 << "  f2=" << f2 << endl;

  // concatinate MyString by operator+
  n1 += f1;                     //"SatoAkira"
  MyString fn2 = n2 + " " + f2; //"Sato Toshio"

  // display concatinated string
  cout << "n1 is " << n1 << endl;
  cout << "fn2 is " << fn2 << endl;

  // test equivalency of string
  if(n1==n2) cout << "n1 and n2 are the same" << endl;
#endif
  if(f1==f2) cout << "f1 and f2 are the same" << endl;

  return(0);
} // automatic objects are destroyed

