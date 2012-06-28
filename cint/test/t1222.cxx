/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <iostream>
using namespace std;

using namespace std;

typedef 
enum { TEST1 = 5,
               TEST2,
               TEST3
               } test_values;

int main( void )
{
  cout << "enum TEST1 = " << TEST1 << endl;
  cout << "enum TEST2 = " << TEST2 << endl;
  cout << "enum TEST3 = " << TEST3 << endl;

  return 0;
}

