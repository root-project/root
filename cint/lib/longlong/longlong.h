/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/longlong/longlong.h
 ************************************************************************
 * Description:
 *  Support 'long long' 64bit integer in 32bit architecture
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for non-commercial purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__LONGLONG_H
#define G__LONGLONG_H

#define IOS

#ifdef IOS
#include <iostream.h>
//using namespace std;
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

typedef _int64 G__int64;

/**************************************************************************
* LINUX
**************************************************************************/
#elif defined(__linux__)

#define G__NODIV
#define G__NOMOD
typedef long long G__int64;

/**************************************************************************
* OTHER
**************************************************************************/
#else

#define G__NODIV
#define G__NOMOD
typedef long long G__int64;

#endif


/************************************************************************
* long long definition
* class G__longlong is renamed as 'long long' in cint body
************************************************************************/

class G__longlong {
 public:
  // constructor
#ifndef __CINT__
  G__longlong(G__int64 x=0) { dat=x; }
#else
  G__longlong(long l=0) { dat = (G__int64)l; }
#endif
#if 0
  G__longlong(long l) { dat = (G__int64)l; }
  G__longlong(int l) { dat = (G__int64)l; }
  G__longlong(short l) { dat = (G__int64)l; }
  G__longlong(char l) { dat = (G__int64)l; }
#endif
  G__longlong(const G__longlong& x) { dat=x.dat; }
  ~G__longlong() {  }

  // conversion operator
#ifndef G__OLDIMPLEMENTATION1144
  operator long() { return((long)dat); }
  operator int() { return((int)dat); }
#endif
  operator double() { return((double)dat); }

  // unary operators
  G__longlong& operator++() { ++dat; return(*this); }
  G__longlong operator++(int dmy) { G__longlong c(dat++); return(c); }
  G__longlong& operator--() { --dat; return(*this); }
  G__longlong operator--(int dmy) { G__longlong c(dat--); return(c); }

  // assignment operators
#ifndef G__OLDIMPLEMENTATION1144
  G__longlong& operator=(const long x) { dat=(G__int64)x; return(*this); }
#endif
  G__longlong& operator=(const G__longlong& x) { dat=x.dat; return(*this); }
  G__longlong& operator+=(const G__longlong& x) { dat+=x.dat; return(*this); }
  G__longlong& operator-=(const G__longlong& x) { dat-=x.dat; return(*this); }
  G__longlong& operator*=(const G__longlong& x) { dat*=x.dat; return(*this); }
#if 0
  G__longlong& operator/=(const G__longlong& x) { dat/=x.dat; return(*this); }
  G__longlong& operator%=(const G__longlong& x) { dat%=x.dat; return(*this); }
  G__longlong& operator&=(const G__longlong& x) { dat&=x.dat; return(*this); }
#endif
  G__longlong& operator|=(const G__longlong& x) { dat|=x.dat; return(*this); }
  G__longlong& operator<<=(const G__longlong& x) { dat<<=x.dat; return(*this);}
  G__longlong& operator>>=(const G__longlong& x) { dat>>=x.dat; return(*this);}

  // binary operators
  friend G__longlong operator+(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator-(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator*(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator/(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator%(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator&(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator|(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator<<(const G__longlong& a,const G__longlong& b);
  friend G__longlong operator>>(const G__longlong& a,const G__longlong& b);

  friend int operator&&(const G__longlong& a,const G__longlong& b);
  friend int operator||(const G__longlong& a,const G__longlong& b);
  friend int operator<(const G__longlong& a,const G__longlong& b);
  friend int operator>(const G__longlong& a,const G__longlong& b);
  friend int operator<=(const G__longlong& a,const G__longlong& b);
  friend int operator>=(const G__longlong& a,const G__longlong& b);
  friend int operator!=(const G__longlong& a,const G__longlong& b);
  friend int operator==(const G__longlong& a,const G__longlong& b);

#ifdef IOS
  friend ostream& operator<<(ostream& ost,const G__longlong& a);
  friend istream& operator>>(istream& ist,const G__longlong& a);
#endif

 private: 
#ifndef __CINT__
  G__int64 dat;
#endif
};

inline G__longlong operator+(const G__longlong& a,const G__longlong& b){
  G__longlong c(a.dat+b.dat);
  return(c);
}
inline G__longlong operator-(const G__longlong& a,const G__longlong& b){
  G__longlong c(a.dat-b.dat);
  return(c);
}
inline G__longlong operator*(const G__longlong& a,const G__longlong& b){
  G__longlong c(a.dat*b.dat);
  return(c);
}
inline G__longlong operator/(const G__longlong& a,const G__longlong& b){
#ifndef G__NODIV
  G__longlong c(a.dat/b.dat);
#else
  G__longlong c;
#endif
  return(c);
}
inline G__longlong operator%(const G__longlong& a,const G__longlong& b){
#ifndef G__NOMOD
  G__longlong c(a.dat%b.dat);
#else
  G__longlong c;
#endif
  return(c);
}
inline G__longlong operator&(const G__longlong& a,const G__longlong& b){
  G__longlong c(a.dat&b.dat);
  return(c);
}
inline G__longlong operator|(const G__longlong& a,const G__longlong& b){
  G__longlong c(a.dat|b.dat);
  return(c);
}

inline G__longlong operator<<(const G__longlong& a,const G__longlong& b){
  G__longlong c(a.dat<<b.dat);
  return(c);
}
inline G__longlong operator>>(const G__longlong& a,const G__longlong& b){
  G__longlong c(a.dat>>b.dat);
  return(c);
}

inline int operator&&(const G__longlong& a,const G__longlong& b){
  return(a.dat&&b.dat);
}
inline int operator||(const G__longlong& a,const G__longlong& b){
  return(a.dat||b.dat);
}
inline int operator<(const G__longlong& a,const G__longlong& b){
  return(a.dat<b.dat);
}
inline int operator>(const G__longlong& a,const G__longlong& b){
  return(a.dat>b.dat);
}
inline int operator<=(const G__longlong& a,const G__longlong& b){
  return(a.dat<=b.dat);
}
inline int operator>=(const G__longlong& a,const G__longlong& b){
  return(a.dat>=b.dat);
}
inline int operator!=(const G__longlong& a,const G__longlong& b){
  return(a.dat!=b.dat);
}
inline int operator==(const G__longlong& a,const G__longlong& b){
  return(a.dat==b.dat);
}

#ifdef IOS
inline ostream& operator<<(ostream& ost,const G__longlong& a) {
  //long *upper = (long*)(&a+1);
  long *lower = (long*)&a;
  ost << *lower ;
  return(ost);
}

inline istream& operator>>(istream& ist,const G__longlong& a) {
  //long *upper = (long*)(&a+1);
  long *lower = (long*)&a;
  ist >> *lower;
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
