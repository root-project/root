/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/longlong/longdbl.h
 ************************************************************************
 * Description:
 *  Support 'long double' 
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for non-commercial purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__LONGDOUBLE_H
#define G__LONGDOUBLE_H

#ifndef IOS
#define IOS
#endif

#ifdef IOS
#ifdef G__NEWSTDHEADER
#include <iostream>
#else
#include <iostream.h>
#endif
#if !defined(__hpux) && !defined(_MSC_VER)
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
  ~G__longdouble() {  }

  // conversion operator
  operator double() { return((double)dat); }

  // unary operators
  G__longdouble& operator++() { ++dat; return(*this); }
  G__longdouble operator++(int dmy) { G__longdouble c(dat++); return(c); }
  G__longdouble& operator--() { --dat; return(*this); }
  G__longdouble operator--(int dmy) { G__longdouble c(dat--); return(c); }

  // assignment operators
  G__longdouble& operator=(const double x) {dat=(G__double92)x;return(*this);}
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

 private: 
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


#ifdef __MAKECINT__
/*
#undef G__REGEXP
#undef G__SHAREDLIB
#undef G__OSFDLL
#pragma eval G__deleteglobal("G__REGEXP");
#pragma eval G__deleteglobal("G__SHAREDLIB");
#pragma eval G__deleteglobal("G__OSFDLL");
#pragma link off global G__REGEXP;
#pragma link off global G__SHAREDLIB;
#pragma link off global G__OSFDLL;
*/
#endif


#endif /* G__LONGLONG_H */
