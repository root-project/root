/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file stdcxxfunc.h
 ************************************************************************
 * Description:
 *  Stub file for making ANSI/ISO C++ standard structs
 ************************************************************************
 * Copyright(c) 2001~2002,   Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__STDCXXFUNC
#define G__STDCXXFUNC


#if 0
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <locale.h>
#endif

#include <ctime>

#if (defined(__sgi) && !defined(__GNUC__) && !defined(__KCC)) || (defined(__alpha) && !defined(__GNUC__))
#include <math.h>
#else
#include <cmath>
#endif
#ifdef __hpux
namespace std {}
#endif

using namespace std;

#ifdef __MAKECINT__

//long  abs  (long);
//ldiv_t div(long, long);

float abs  (float);
float acos (float);
float asin (float);
float atan (float);
float atan2(float, float);
float ceil (float);
float cos  (float);
float cosh (float);
float exp  (float);
float fabs (float);
float floor(float);
float fmod (float, float);
float frexp(float, int*);
float ldexp(float, int);
float log  (float);
float log10(float);
//float modf (float, float*);
float pow  (float, float);
float pow  (float, int);
float sin  (float);
float sinh (float);
float sqrt (float);
float tan  (float);
float tanh (float);

double abs(double);            // fabs()
double pow(double, int);

size_t strftime(char* ptr, size_t maxsize, const char* fmt, const struct tm* time);

#if !defined(G__SUN) && !defined(G__AIX)
long double abs  (long double);
long double acos (long double);
long double asin (long double);
long double atan (long double);
long double atan2(long double, long double);
long double ceil (long double);
long double cos  (long double);
long double cosh (long double);
long double exp  (long double);
long double fabs (long double);
long double floor(long double);
long double fmod (long double, long double);
long double frexp(long double, int*);
long double ldexp(long double, int);
long double log  (long double);
long double log10(long double);
//long double modf (long double, long double*);
long double pow  (long double, long double);
long double pow  (long double, int);
long double sin  (long double);
long double sinh (long double);
long double sqrt (long double);
long double tan  (long double);
long double tanh (long double);
#endif

#endif

#endif
