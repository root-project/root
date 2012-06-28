/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
* eular.c
*
* array pre-compiled class library is needed to run this demo program.
*************************************************************************/
#include <array.h>
const complex j=complex(0,1);
const double PI=3.141592;

main() {
  array x=array(-2*PI , 2*PI , 100 ); // start,stop,npoint
  plot << "Eular's Law" << x << exp(x*j) << "exp(x*j)" << endl ;
}
