/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* Array.C
*
* Array class instanciation
*
**************************************************************************/
#include "Array.h"
#include "Complex.C"

using namespace std;

/**************************************************************************
* int dummy
**************************************************************************/
Array<int> exp(Array<int>& a)
{
  cerr << "exp(Array<int>) not supported\n" ;
  return(a);
}

#ifdef __GNUC__
Array<int> abs(Array<int>& a)
{
  a.setdefaultsize(a.n);
  Array<int> c;
  for(int i=0;i<a.n;i++) c[i] = (int)fabs(a[i]);
  return(c);
}

Array<Complex> exp(Array<Complex>& a)
{
  a.setdefaultsize(a.n);
  Array<Complex> c;
  for(int i=0;i<a.n;i++) c[i] = exp(a[i]);
  return(c);
}

Array<double> abs(Array<double>& a)
{
  a.setdefaultsize(a.n);
  Array<double> c;
  for(int i=0;i<a.n;i++) c[i] = (double)fabs(a[i]);
  return(c);
}

Array<Complex> abs(Array<Complex>& a)
{
  a.setdefaultsize(a.n);
  Array<Complex> c;
  for(int i=0;i<a.n;i++) c[i] = fabs(a[i]);
  return(c);
}

Array<double> exp(Array<double>& a)
{
  a.setdefaultsize(a.n);
  Array<double> c;
  for(int i=0;i<a.n;i++) c[i] = (double)exp(a[i]);
  return(c);
}
#endif

#ifdef G__NOSTATICMEMBER
int G__defaultsize = 100;
#endif
