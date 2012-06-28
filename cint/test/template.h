/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// #ifndef G__TEMPLATE_H

// #define G__TEMPLATE_H

#include <iostream>
using namespace std;

#ifndef __hpux
#define MEMFUNCTMPLT
#endif

template<class T,int SZ> class ary {
 public:
  T body[SZ];
  int sz;
 public:
#if defined(MEMFUNCTMPLT) || defined(__MAKECINT__)
  ary(T os=0,T gain=0) { 
    int i;
    sz = SZ; 
    for(i=0;i<SZ;i++) body[i] = os + i*gain;
  }
  ary& operator=(const ary& a) {
    int i;
    for(i=0;i<SZ;i++) body[i]=a.body[i];
    return(*this);
  }
  ary operator+(ary& a) {
    ary result;
    int i;
    for(i=0;i<SZ;i++) result.body[i]=body[i]+a.body[i];
    return(result);
  }
  void disp() {
    int i;
    cout << "display\n";
    for(i=0;i<SZ;i++) cout << "body[" << i << "]=" << body[i] << "\n";
  }
#else
  ary(T os=0,T gain=0) ; 
  ary& operator=(ary& a) ;
  ary operator+(ary&) ;
  void disp() ;
#endif
};

void test(void);

#ifdef __MAKECINT__
typedef ary<double,10> G__dummy;
#endif

#if !defined(MEMFUNCTMPLT) && !defined(__MAKECINT__)
template<class T,int SZ> ary<T,SZ>::ary(T os,T gain) {
    int i;
    sz = SZ; 
    for(i=0;i<SZ;i++) body[i] = os + i*gain;
}
template<class T,int SZ> ary<T,SZ>& ary<T,SZ>::operator=(ary<T,SZ>& a) {
    int i;
    for(i=0;i<SZ;i++) body[i]=a.body[i];
    return(*this);
}
template<class T,int SZ>  ary<T,SZ> ary<T,SZ>::operator+(ary<T,SZ>& a) {
    ary<T,SZ> result;
    int i;
    for(i=0;i<SZ;i++) result.body[i]=body[i]+a.body[i];
    return(result);
  }
template<class T,int SZ> void ary<T,SZ>::disp() {
    int i;
    cout << "display\n";
    for(i=0;i<SZ;i++) cout << "body[" << i << "]=" << body[i] << "\n";
  }
#endif

//#endif

void test(void)
{
  ary<double,10> a(100,10),b(5,1),c;
  a.disp();
  b.disp();
  c = a+b;
  c.disp();
}
