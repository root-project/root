/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* Complex.h
*
*
**************************************************************************/

#ifndef COMPLEX_H
#define COMPLEX_H

#include <stdio.h>
#include <math.h>

struct Complex {
  double re,im;
};

/* makecint requests ANSI style function header to get parameter information.
* while K&R C compiler does not accept it. 
* You have to use #ifdef __MAKECINT__ to work around the difference.
* This handling is only needed in header files because makecint or 
* 'cint -c-2' only reads header file for getting the interface. */

extern struct Complex j;

#ifdef __MAKECINT__

/* ANSI style function header is needed for makecint or cint -c-2
* If you use __P() macro for parameters, you must give -p option like
* cint -c-2 -p -D__MAKECINT__ xxx.h  */

void ComplexInit(void);
void ComplexSet(struct Complex *pa,double rein,double imin);
struct Complex ComplexAdd(struct Complex a,struct Complex b);
struct Complex ComplexMultiply(struct Complex a,struct Complex b);
struct Complex ComplexExp(struct Complex a);
double ComplexAbs(struct Complex a);
double ComplexReal(struct Complex a);
double ComplexImag(struct Complex a);
void ComplexDisplay(struct Complex a);
double add(double a,double b);

#else

/* This part is used by K&R C Compler */

void ComplexInit();
void ComplexSet();
struct Complex ComplexAdd();
struct Complex ComplexMultiply();
struct Complex ComplexExp();
double ComplexAbs();
double ComplexReal();
double ComplexImag();
void ComplexDisplay();
#define add(a,b)  (a+b)

#endif

#endif
