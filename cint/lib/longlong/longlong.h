/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/longlong/longlong.h
 ************************************************************************
 * Description:
 *  Support 'long long' 64bit integer in 32bit architecture
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
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
* 64bit platforms
**************************************************************************/
#elif defined(G__64BITLONG)

#define G__NODIV
#define G__NOMOD
typedef long G__int64;
typedef unsigned long G__uint64;


/**************************************************************************
* WIN32
**************************************************************************/
#elif defined(_WIN32)||defined(_WINDOWS)||defined(_Windows)||defined(_WINDOWS_)

#if defined(_MSC_VER)
typedef _int64 G__int64;
typedef unsigned _int64 G__uint64;
#elif defined(__BCPLUSPLUS__)
typedef __int64 G__int64;
typedef unsigned _int64 G__uint64;
#else
typedef long long G__int64;
typedef unsgined long long G__uint64;
#endif

/**************************************************************************
* LINUX
**************************************************************************/
#elif defined(__linux__)

#define G__NODIV
#define G__NOMOD
typedef long long G__int64;
typedef unsigned long long G__uint64;

/**************************************************************************
* OTHER
**************************************************************************/
#else

#define G__NODIV
#define G__NOMOD
typedef long long G__int64;
typedef unsigned long long G__uint64;

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
#endif
  G__longlong& operator&=(const G__longlong& x) { dat&=x.dat; return(*this); }
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
  friend istream& operator>>(istream& ist,G__longlong& a);
#endif

  //private: 
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
#ifndef G__OLDIMPLEMENTATION1686
  char buf[50];
  sprintf(buf,"%lld",a.dat);
  ost << buf;
#else
  //long *upper = (long*)(&a+1);
  long *lower = (long*)&a;
  ost << *lower ;
#endif
  return(ost);
}

inline istream& operator>>(istream& ist,G__longlong& a) {
#ifndef G__OLDIMPLEMENTATION1686
  char buf[50];
  ist >> buf;
  sscanf(buf,"%lld",&a.dat);
#else
  //long *upper = (long*)(&a+1);
  long *lower = (long*)&a;
  ist >> *lower;
#endif
  return(ist);
}
#endif

int G__ateval(const G__longlong& a) {
  fprintf(stdout,"(long long)%lld\n",a.dat);
  return(1);
}


/************************************************************************
* unsigned long long definition
* class G__ulonglong is renamed as 'long long' in cint body
************************************************************************/

class G__ulonglong {
 public:
  // constructor
#ifndef __CINT__
  G__ulonglong(G__uint64 x=0) { dat=x; }
#else
  G__ulonglong(long l=0) { dat = (G__uint64)l; }
#endif
#if 0
  G__ulonglong(long l) { dat = (G__uint64)l; }
  G__ulonglong(int l) { dat = (G__uint64)l; }
  G__ulonglong(short l) { dat = (G__uint64)l; }
  G__ulonglong(char l) { dat = (G__uint64)l; }
#endif
  G__ulonglong(const G__ulonglong& x) { dat=x.dat; }
  ~G__ulonglong() {  }

  // conversion operator
#ifndef G__OLDIMPLEMENTATION1144
  operator long() { return((long)dat); }
  operator int() { return((int)dat); }
#endif
  operator double() { return((double)dat); }

  // unary operators
  G__ulonglong& operator++() { ++dat; return(*this); }
  G__ulonglong operator++(int dmy) { G__ulonglong c(dat++); return(c); }
  G__ulonglong& operator--() { --dat; return(*this); }
  G__ulonglong operator--(int dmy) { G__ulonglong c(dat--); return(c); }

  // assignment operators
#ifndef G__OLDIMPLEMENTATION1144
  G__ulonglong& operator=(const long x) { dat=(G__uint64)x; return(*this); }
#endif
  G__ulonglong& operator=(const G__ulonglong& x) { dat=x.dat; return(*this); }
  G__ulonglong& operator+=(const G__ulonglong& x) { dat+=x.dat; return(*this); }
  G__ulonglong& operator-=(const G__ulonglong& x) { dat-=x.dat; return(*this); }
  G__ulonglong& operator*=(const G__ulonglong& x) { dat*=x.dat; return(*this); }
#if 0
  G__ulonglong& operator/=(const G__ulonglong& x) { dat/=x.dat; return(*this); }
  G__ulonglong& operator%=(const G__ulonglong& x) { dat%=x.dat; return(*this); }
#endif
  G__ulonglong& operator&=(const G__ulonglong& x) { dat&=x.dat; return(*this); }
  G__ulonglong& operator|=(const G__ulonglong& x) { dat|=x.dat; return(*this); }
  G__ulonglong& operator<<=(const G__ulonglong& x) { dat<<=x.dat; return(*this);}
  G__ulonglong& operator>>=(const G__ulonglong& x) { dat>>=x.dat; return(*this);}

  // binary operators
  friend G__ulonglong operator+(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator-(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator*(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator/(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator%(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator&(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator|(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator<<(const G__ulonglong& a,const G__ulonglong& b);
  friend G__ulonglong operator>>(const G__ulonglong& a,const G__ulonglong& b);

  friend int operator&&(const G__ulonglong& a,const G__ulonglong& b);
  friend int operator||(const G__ulonglong& a,const G__ulonglong& b);
  friend int operator<(const G__ulonglong& a,const G__ulonglong& b);
  friend int operator>(const G__ulonglong& a,const G__ulonglong& b);
  friend int operator<=(const G__ulonglong& a,const G__ulonglong& b);
  friend int operator>=(const G__ulonglong& a,const G__ulonglong& b);
  friend int operator!=(const G__ulonglong& a,const G__ulonglong& b);
  friend int operator==(const G__ulonglong& a,const G__ulonglong& b);

#ifdef IOS
  friend ostream& operator<<(ostream& ost,const G__ulonglong& a);
  friend istream& operator>>(istream& ist,G__ulonglong& a);
#endif

  //private: 
#ifndef __CINT__
  G__uint64 dat;
#endif
};

inline G__ulonglong operator+(const G__ulonglong& a,const G__ulonglong& b){
  G__ulonglong c(a.dat+b.dat);
  return(c);
}
inline G__ulonglong operator-(const G__ulonglong& a,const G__ulonglong& b){
  G__ulonglong c(a.dat-b.dat);
  return(c);
}
inline G__ulonglong operator*(const G__ulonglong& a,const G__ulonglong& b){
  G__ulonglong c(a.dat*b.dat);
  return(c);
}
inline G__ulonglong operator/(const G__ulonglong& a,const G__ulonglong& b){
#ifndef G__NODIV
  G__ulonglong c(a.dat/b.dat);
#else
  G__ulonglong c;
#endif
  return(c);
}
inline G__ulonglong operator%(const G__ulonglong& a,const G__ulonglong& b){
#ifndef G__NOMOD
  G__ulonglong c(a.dat%b.dat);
#else
  G__ulonglong c;
#endif
  return(c);
}
inline G__ulonglong operator&(const G__ulonglong& a,const G__ulonglong& b){
  G__ulonglong c(a.dat&b.dat);
  return(c);
}
inline G__ulonglong operator|(const G__ulonglong& a,const G__ulonglong& b){
  G__ulonglong c(a.dat|b.dat);
  return(c);
}

inline G__ulonglong operator<<(const G__ulonglong& a,const G__ulonglong& b){
  G__ulonglong c(a.dat<<b.dat);
  return(c);
}
inline G__ulonglong operator>>(const G__ulonglong& a,const G__ulonglong& b){
  G__ulonglong c(a.dat>>b.dat);
  return(c);
}

inline int operator&&(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat&&b.dat);
}
inline int operator||(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat||b.dat);
}
inline int operator<(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat<b.dat);
}
inline int operator>(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat>b.dat);
}
inline int operator<=(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat<=b.dat);
}
inline int operator>=(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat>=b.dat);
}
inline int operator!=(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat!=b.dat);
}
inline int operator==(const G__ulonglong& a,const G__ulonglong& b){
  return(a.dat==b.dat);
}

#ifdef IOS
inline ostream& operator<<(ostream& ost,const G__ulonglong& a) {
#ifndef G__OLDIMPLEMENTATION1686
  char buf[50];
  sprintf(buf,"%lld",a.dat);
  ost << buf;
#else
  //long *upper = (long*)(&a+1);
  long *lower = (long*)&a;
  ost << *lower ;
#endif
  return(ost);
}

inline istream& operator>>(istream& ist,G__ulonglong& a) {
#ifndef G__OLDIMPLEMENTATION1686
  char buf[50];
  ist >> buf;
  sscanf(buf,"%lld",&a.dat);
#else
  //long *upper = (long*)(&a+1);
  long *lower = (long*)&a;
  ist >> *lower;
#endif
  return(ist);
}
#endif

int G__ateval(const G__ulonglong& a) {
  fprintf(stdout,"(unsigned long long)%llu\n",a.dat);
  return(1);
}


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
