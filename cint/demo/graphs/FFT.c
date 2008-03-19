/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/******************************************************
* sin.c
*
* array pre-compiled class library is needed to run 
* this demo program.
******************************************************/
#include <array.h>
#include <constants.h>
#include <fft.h>
#define NPOINT 1024
main() {
  double width = 10*2*PI*(NPOINT-1)/NPOINT;
  array x=array(0 , width , NPOINT ),y; // start,stop,npoint
  y = sin(x) + 0.5*sin(x*3);
  plot << x << y << "\n" ;

  array freq,fftout;
  spectrum << x << y >> freq >> fftout >> '\n';
  plot << freq << fftout << '\n';

  carray cfftout;
  fft << x << y >> freq >> cfftout >> endl;
  plot << freq << cfftout << endl;
}
