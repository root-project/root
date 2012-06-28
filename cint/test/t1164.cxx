/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/



#include <stdio.h>

class CTest
{
public:
  CTest( void ) { return; };
  ~CTest() { return; };
  
  void Set( int x ) { printf( "Set int = %d\n", x ); };
  void Set( long x ) { printf( "Set long = %ld\n", x ); };
  void Set( double x ) { printf( "Set double = %f\n", x ); };
};

int main ( void ) 
{
  CTest tc;
  
  printf( "Expecting set int -> " );
  tc.Set( 10 );   // expect int (ok)
  
  printf( "Expecting set double -> " );
  tc.Set( 20.0 ); // expect double (ok)
  
  printf( "Expecting set int -> " );
  int ival = 10;
  tc.Set( ival );   // expect int (ok)
  
  printf( "Expecting set int -> " );
  short sval = 10;
  tc.Set( sval );   // expect int (ok)
  
  // ERROR with set of double
  printf( "Expecting set int -> " );
  long lval = 13;
  tc.Set( lval );   // expect int (ERROR getting double)
  
  printf( "Expecting set double -> " );
  float fval = 10.0;
  tc.Set( fval );   // expect double (ok) 
  
  return 0;
}







