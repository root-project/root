/********************************************************************
* longif.h
********************************************************************/
#ifdef __CINT__
#error longif.h/C is only for compilation. Abort cint.
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define G__ANSIHEADER
#define G__DICTIONARY
#include "G__ci.h"
extern "C" {
extern void G__cpp_setup_tagtablelongif();
extern void G__cpp_setup_inheritancelongif();
extern void G__cpp_setup_typetablelongif();
extern void G__cpp_setup_memvarlongif();
extern void G__cpp_setup_globallongif();
extern void G__cpp_setup_memfunclongif();
extern void G__cpp_setup_funclongif();
extern void G__set_cpp_environmentlongif();
}


//#include "longlong.h"
/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/longlong/longlong.h
 ************************************************************************
 * Description:
 *  Support 'long long' 64bit integer in 32bit architecture
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto (MXJ02154@niftyserve.or.jp)
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
namespace std {} using namespace std;
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
* 64bit platforms
**************************************************************************/
#elif defined(__GNUC__) && (__GNUC__>=3)

typedef long long G__int64;
typedef unsigned long long G__uint64;


/**************************************************************************
* WIN32
**************************************************************************/
#elif defined(_WIN32)||defined(_WINDOWS)||defined(_Windows)||defined(_WINDOWS_)

#if defined(_MSC_VER)
typedef __int64 G__int64;
typedef unsigned __int64 G__uint64;
#elif defined(__BCPLUSPLUS__)
typedef __int64 G__int64;
typedef unsigned __int64 G__uint64;
#else
typedef long long G__int64;
typedef unsigned long long G__uint64;
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


#ifndef G__OLDIMPLEMENTATION1878
#ifndef __CINT__
/**************************************************************************
* G__strtoll, G__strtoull
**************************************************************************/
#include <ctype.h>
#include <errno.h>

#ifndef ULONG_LONG_MAX
//#define       ULONG_LONG_MAX  ((G__uint64)(~0LL))
#define       ULONG_LONG_MAX  (~((G__uint64)0))
#endif

#ifndef LONG_LONG_MAX
#define       LONG_LONG_MAX   ((G__int64)(ULONG_LONG_MAX >> 1))
#endif

#ifndef LONG_LONG_MIN
#define       LONG_LONG_MIN   ((G__int64)(~LONG_LONG_MAX))
#endif

/*
 * Convert a string to a long long integer.
 *
 * Ignores `locale' stuff.  Assumes that the upper and lower case
 * alphabets and digits are each contiguous.
 */
G__int64 G__strtoll(const char *nptr,char **endptr, register int base) {
   register const char *s = nptr;
   register G__uint64 acc;
   register int c;
   register G__uint64 cutoff;
   register int neg = 0, any, cutlim;

   /*
    * Skip white space and pick up leading +/- sign if any.
    * If base is 0, allow 0x for hex and 0 for octal, else
    * assume decimal; if base is already 16, allow 0x.
    */
   do {
      c = *s++;
   }
   while (isspace(c));
   if (c == '-') {
      neg = 1;
      c = *s++;
   } else if (c == '+')
      c = *s++;
   if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) {
      c = s[1];
      s += 2;
      base = 16;
   }
   if (base == 0)
      base = c == '0' ? 8 : 10;

   /*
    * Compute the cutoff value between legal numbers and illegal
    * numbers.  That is the largest legal value, divided by the
    * base.  An input number that is greater than this value, if
    * followed by a legal input character, is too big.  One that
    * is equal to this value may be valid or not; the limit
    * between valid and invalid numbers is then based on the last
    * digit.  For instance, if the range for long longs is
    * [-2147483648..2147483647] and the input base is 10,
    * cutoff will be set to 214748364 and cutlim to either
    * 7 (neg==0) or 8 (neg==1), meaning that if we have accumulated
    * a value > 214748364, or equal but the next digit is > 7 (or 8),
    * the number is too big, and we will return a range error.
    *
    * Set any if any `digits' consumed; make it negative to indicate
    * overflow.
    */
   cutoff = neg ? -(G__uint64) LONG_LONG_MIN : LONG_LONG_MAX;
   cutlim = cutoff % (G__uint64) base;
   cutoff /= (G__uint64) base;
   for (acc = 0, any = 0;; c = *s++) {
      if (isdigit(c))
         c -= '0';
      else if (isalpha(c))
         c -= isupper(c) ? 'A' - 10 : 'a' - 10;
      else
         break;
      if (c >= base)
         break;
      if (any < 0 || acc > cutoff || acc == cutoff && c > cutlim)
         any = -1;
      else {
         any = 1;
         acc *= base;
         acc += c;
      }
   }
   if (any < 0) {
      acc = neg ? LONG_LONG_MIN : LONG_LONG_MAX;
      errno = ERANGE;
   } else if (neg)
      acc = -acc;
   if (endptr != 0)
      *endptr = (char *) (any ? s - 1 : nptr);
   return (acc);
}

/*
 * Convert a string to an unsigned long integer.
 *
 * Ignores `locale' stuff.  Assumes that the upper and lower case
 * alphabets and digits are each contiguous.
 */
G__uint64 G__strtoull(const char *nptr, char **endptr, register int base) {
   register const char *s = nptr;
   register G__uint64 acc;
   register int c;
   register G__uint64 cutoff;
   register int neg = 0, any, cutlim;

   /*
    * See strtoll for comments as to the logic used.
    */
   do {
      c = *s++;
   }
   while (isspace(c));
   if (c == '-') {
      neg = 1;
      c = *s++;
   } else if (c == '+')
      c = *s++;
   if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) {
      c = s[1];
      s += 2;
      base = 16;
   }
   if (base == 0)
      base = c == '0' ? 8 : 10;
   cutoff =
       (G__uint64) ULONG_LONG_MAX / (G__uint64) base;
   cutlim =
       (G__uint64) ULONG_LONG_MAX % (G__uint64) base;
   for (acc = 0, any = 0;; c = *s++) {
      if (isdigit(c))
         c -= '0';
      else if (isalpha(c))
         c -= isupper(c) ? 'A' - 10 : 'a' - 10;
      else
         break;
      if (c >= base)
         break;
      if (any < 0 || acc > cutoff || (acc == cutoff && c > cutlim))
         any = -1;
      else {
         any = 1;
         acc *= base;
         acc += c;
      }
   }
   if (any < 0) {
      acc = ULONG_LONG_MAX;
      errno = ERANGE;
   } else if (neg)
      acc = -acc;
   if (endptr != 0)
      *endptr = (char *) (any ? s - 1 : nptr);
   return (acc);
}
#endif /* __CINT__ */
#endif /* 1878 */

/************************************************************************
* long long definition
* class G__longlong is renamed as 'long long' in cint body
************************************************************************/
class G__ulonglong;

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
  G__longlong(const G__ulonglong& x) ;
#if 1
  G__longlong(const char* s) { dat=G__strtoll(s,NULL,10); }
#endif
  ~G__longlong() {  }

  // conversion operator
#ifndef G__OLDIMPLEMENTATION1144
  operator long() { return((long)dat); }
  operator int() { return((int)dat); }
#endif
  operator double() { return((double)dat); }

  // unary operators
  G__longlong& operator++() { ++dat; return(*this); }
  G__longlong operator++(int) { G__longlong c(dat++); return(c); }
  G__longlong& operator--() { --dat; return(*this); }
  G__longlong operator--(int) { G__longlong c(dat--); return(c); }

  // assignment operators
#ifndef G__OLDIMPLEMENTATION1144
  G__longlong& operator=(long x) { dat=(G__int64)x; return(*this); }
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
  fprintf(stderr,"Limitation: operator/ is deactivated for 'long long'. Delete G__NODIV in $CINTSYSDIR/src/longif3.h, longif.h and $CINTSYSDIR/lib/longlong/longlong.h to activate.\n");
#endif
  return(c);
}
inline G__longlong operator%(const G__longlong& a,const G__longlong& b){
#ifndef G__NOMOD
  G__longlong c(a.dat%b.dat);
#else
  G__longlong c;
  fprintf(stderr,"Limitation: operator%% is deactivated for 'long long'. Delete G__NOMOD in $CINTSYSDIR/src/longif3.h, longif.h and $CINTSYSDIR/lib/longlong/longlong.h to activate.\n");
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
  G__ulonglong(unsigned long l=0) { dat = (G__uint64)l; }
#endif
#if 0
  G__ulonglong(long l) { dat = (G__uint64)l; }
  G__ulonglong(int l) { dat = (G__uint64)l; }
  G__ulonglong(short l) { dat = (G__uint64)l; }
  G__ulonglong(char l) { dat = (G__uint64)l; }
#endif
  G__ulonglong(const G__ulonglong& x) { dat=x.dat; }
#if 1
  G__ulonglong(const G__longlong& x) { dat=(G__int64)x.dat; }
  G__ulonglong(const char* s) { dat=G__strtoull(s,NULL,10); }
#endif
  ~G__ulonglong() {  }

  // conversion operator
#ifndef G__OLDIMPLEMENTATION1144
  operator long() { return((long)dat); }
  operator int() { return((int)dat); }
#endif
#if defined(G__VISUAL)
#else
  //operator double() { return((double)dat); }
#endif

  // unary operators
  G__ulonglong& operator++() { ++dat; return(*this); }
  G__ulonglong operator++(int) { G__ulonglong c(dat++); return(c); }
  G__ulonglong& operator--() { --dat; return(*this); }
  G__ulonglong operator--(int) { G__ulonglong c(dat--); return(c); }

  // assignment operators
#ifndef G__OLDIMPLEMENTATION1144
  G__ulonglong& operator=(long x) { dat=(G__uint64)x; return(*this); }
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
  fprintf(stderr,"Limitation: operator/ is deactivated for 'unsigned long long'. Delete G__NODIV in $CINTSYSDIR/src/longif3.h, longif.h and $CINTSYSDIR/lib/longlong/longlong.h to activate.\n");
#endif
  return(c);
}
inline G__ulonglong operator%(const G__ulonglong& a,const G__ulonglong& b){
#ifndef G__NOMOD
  G__ulonglong c(a.dat%b.dat);
#else
  G__ulonglong c;
  fprintf(stderr,"Limitation: operator%% is deactivated for 'unsigned long long'. Delete G__NOMOD in $CINTSYSDIR/src/longif3.h, longif.h and $CINTSYSDIR/lib/longlong/longlong.h to activate.\n");
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

inline G__longlong::G__longlong(const G__ulonglong& x) { dat=(G__int64)x.dat; }

#ifdef IOS
inline ostream& operator<<(ostream& ost,const G__ulonglong& a) {
#ifndef G__OLDIMPLEMENTATION1686
  char buf[50];
  sprintf(buf,"%llu",a.dat);
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
  sscanf(buf,"%llu",&a.dat);
#else
  //long *upper = (long*)(&a+1);
  long *lower = (long*)&a;
  ist >> *lower;
#endif
  return(ist);
}
#endif

#include <stdio.h>
void G__printformatll(char* out,const char* fmt,void *p) {
  //long long *pll = (long long*)p;
  G__int64 *pll = (G__int64*)p;
  sprintf(out,fmt,*pll);
}
void G__printformatull(char* out,const char* fmt,void *p) {
  //unsigned long long *pll = (unsigned long long*)p;
  G__uint64 *pll = (G__uint64*)p;
  sprintf(out,fmt,*pll);
}


inline int G__ateval(const G__longlong& a) {
  fprintf(stdout,"(long long)%lld\n",a.dat);
  return(1);
}
inline int G__ateval(const G__ulonglong& a) {
  fprintf(stdout,"(unsigned long long)%llu\n",a.dat);
  return(1);
}
#if 0
int G__ateval(const char* x) {return(0);}
int G__ateval(const void* x) {return(0);}
int G__ateval(double x) {return(0);}
int G__ateval(float x) {return(0);}
int G__ateval(char x) {return(0);}
int G__ateval(short x) {return(0);}
int G__ateval(int x) {return(0);}
int G__ateval(long x) {return(0);}
int G__ateval(unsigned char x) {return(0);}
int G__ateval(unsigned short x) {return(0);}
int G__ateval(unsigned int x) {return(0);}
int G__ateval(unsigned long x) {return(0);}
#endif

#ifdef __MAKECINT__
#ifndef G__LONGLONGTMP
#define G__LONGLONGTMP
#pragma link off global G__LONGLONGTMP;
#endif
#ifdef G__OLDIMPLEMENTATION1912
#pragma link C++ function G__ateval;
#endif
#endif


#endif /* G__LONGLONG_H */

//#include "longdbl.h"
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

//#include "longlong.h"

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
  //long double*pll = (long double*)p;
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

//#include "lib/longlong/longlong.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__longifLN_basic_istreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__longifLN_basic_ostreamlEcharcOchar_traitslEchargRsPgR;
extern G__linked_taginfo G__longifLN_G__ulonglong;
extern G__linked_taginfo G__longifLN_G__longlong;
extern G__linked_taginfo G__longifLN_G__longdouble;

/* STUB derived class for protected member access */
