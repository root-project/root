/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/longlong/longdbl.h
 ************************************************************************
 * Description:
 *  Support 'long double' 
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__LONGDOUBLE_H
#define G__LONGDOUBLE_H

#include "longlong.h"

//#if !defined(__hpux) && !defined(G__HPUX)

#ifndef IOS
#define IOS
#endif

#if (defined(__GNUC__)&&(__GNUC__>=3)) || (defined(_MSC_VER)&&(_MSC_VER>=1300))
#ifndef G__NEWSTDHEADER
#define G__NEWSTDHEADER
#endif
#endif

#ifdef IOS
#ifdef G__NEWSTDHEADER
#include <iostream>
#else
#include <iostream.h>
#endif
#if !defined(__hpux) && !(defined(_MSC_VER) && (_MSC_VER<1200))
using namespace std;
#endif
#endif

/**************************************************************************
* makecint
**************************************************************************/
#if defined(__CINT__)

// #include <bool.h>

/**************************************************************************
* WIN32
**************************************************************************/
#elif defined(_WIN32)||defined(_WINDOWS)||defined(_Windows)||defined(_WINDOWS_)

typedef long double G__double92;

/**************************************************************************
* LINUX
**************************************************************************/
#elif defined(__linux__)

typedef long double G__double92;

/**************************************************************************
* HP-UX
**************************************************************************/
#elif defined(__hpux) || defined(G__HPUX)

typedef double G__double92;

/**************************************************************************
* D.Cussol : Alpha TRU64
**************************************************************************/
#elif defined(__alpha) || defined(G__ALPHA) || defined(R__ALPHA)
				
typedef double G__double92;	

/**************************************************************************
* OTHER
**************************************************************************/
#else

typedef long double G__double92;

#endif


/************************************************************************
* long double definition
* class G__longdouble is renamed as 'long long' in cint body
************************************************************************/

class G__longdouble {
 public:
  // constructor
#ifndef __CINT__
  G__longdouble(G__double92 x=0) { dat=x; }
#else
  G__longdouble(double l=0) { dat = (G__double92)l; }
#endif
#if 0
  G__longdouble(float l) { dat = (G__double92)l; }
  G__longdouble(double l) { dat = (G__double92)l; }
#endif
  G__longdouble(const G__longdouble& x) { dat=x.dat; }
  //G__longdouble(long l=0) { dat = (G__double92)l; }
  G__longdouble(const G__longlong& x) { dat=x.dat; }
#ifndef G__OLDIMPLEMENTATION2007
#if defined(_MSC_VER)&&(_MSC_VER<1310)
  G__longdouble(const G__ulonglong& x) { dat=(G__int64)x.dat; }
#else
  G__longdouble(const G__ulonglong& x) { dat=x.dat; }
#endif
#else
  G__longdouble(const G__ulonglong& x) { dat=x.dat; }
#endif
  ~G__longdouble() {  }

  // conversion operator
  operator double() { return((double)dat); }

  // unary operators
  G__longdouble& operator++() { ++dat; return(*this); }
  G__longdouble operator++(int) { G__longdouble c(dat++); return(c); }
  G__longdouble& operator--() { --dat; return(*this); }
  G__longdouble operator--(int) { G__longdouble c(dat--); return(c); }

  // assignment operators
  G__longdouble& operator=(double x) {dat=(G__double92)x;return(*this);}
  G__longdouble& operator=(const G__longdouble& x) {dat=x.dat;return(*this); }
  G__longdouble& operator+=(const G__longdouble& x) {dat+=x.dat; return(*this); }
  G__longdouble& operator-=(const G__longdouble& x) {dat-=x.dat; return(*this); }
  G__longdouble& operator*=(const G__longdouble& x) {dat*=x.dat; return(*this); }
  G__longdouble& operator/=(const G__longdouble& x) {dat/=x.dat; return(*this); }

  // binary operators
  friend G__longdouble operator+(const G__longdouble& a,const G__longdouble& b);
  friend G__longdouble operator-(const G__longdouble& a,const G__longdouble& b);
  friend G__longdouble operator*(const G__longdouble& a,const G__longdouble& b);
  friend G__longdouble operator/(const G__longdouble& a,const G__longdouble& b);

  friend int operator<(const G__longdouble& a,const G__longdouble& b);
  friend int operator>(const G__longdouble& a,const G__longdouble& b);
  friend int operator<=(const G__longdouble& a,const G__longdouble& b);
  friend int operator>=(const G__longdouble& a,const G__longdouble& b);
  friend int operator!=(const G__longdouble& a,const G__longdouble& b);
  friend int operator==(const G__longdouble& a,const G__longdouble& b);

#ifdef IOS
  friend ostream& operator<<(ostream& ost,const G__longdouble& a);
  friend istream& operator>>(istream& ist,G__longdouble& a);
#endif

  //private: 
#ifndef __CINT__
  G__double92 dat;
#endif
};

inline G__longdouble operator+(const G__longdouble& a,const G__longdouble& b){
  G__longdouble c(a.dat+b.dat);
  return(c);
}
inline G__longdouble operator-(const G__longdouble& a,const G__longdouble& b){
  G__longdouble c(a.dat-b.dat);
  return(c);
}
inline G__longdouble operator*(const G__longdouble& a,const G__longdouble& b){
  G__longdouble c(a.dat*b.dat);
  return(c);
}
inline G__longdouble operator/(const G__longdouble& a,const G__longdouble& b){
  G__longdouble c(a.dat/b.dat);
  return(c);
}

inline int operator<(const G__longdouble& a,const G__longdouble& b){
  return(a.dat<b.dat);
}
inline int operator>(const G__longdouble& a,const G__longdouble& b){
  return(a.dat>b.dat);
}
inline int operator<=(const G__longdouble& a,const G__longdouble& b){
  return(a.dat<=b.dat);
}
inline int operator>=(const G__longdouble& a,const G__longdouble& b){
  return(a.dat>=b.dat);
}
inline int operator!=(const G__longdouble& a,const G__longdouble& b){
  return(a.dat!=b.dat);
}
inline int operator==(const G__longdouble& a,const G__longdouble& b){
  return(a.dat==b.dat);
}

#ifdef IOS
inline ostream& operator<<(ostream& ost,const G__longdouble& a) {
  ost << a.dat;
  return(ost);
}

inline istream& operator>>(istream& ist,G__longdouble& a) {
  ist >> a.dat;
  return(ist);
}
#endif

#ifndef G__OLDIMPLEMENTATION1913
void G__printformatld(char* out,const char* fmt,void *p) {
  //long double*pld = (long double*)p;
  G__double92 *pld = (G__double92*)p;
  sprintf(out,fmt,*pld);
}
#endif

inline int G__ateval(const G__longdouble& a) {
#ifdef IOS
  cout << "(long double)" << a.dat << endl;
#else
  fprintf(stdout,"(long double)%g\n",a.dat);
#endif
  return(1);
}

#ifdef __MAKECINT__
#ifndef G__LONGLONGTMP
#define G__LONGLONGTMP
#pragma link off global G__LONGLONGTMP;
#endif
#ifdef G__OLDIMPLEMENTATION1912
#pragma link C++ function G__ateval;
#endif
#endif

//#endif

#endif /* G__LONGDBL_H */
