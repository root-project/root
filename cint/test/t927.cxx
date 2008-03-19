/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <cstdio>
#include <vector>
#include <list>
#include <typeinfo>
using namespace std;


template<class T>
void func(T dmy) {
  vector<T> x;
  for(int i=0;i<5;i++) {
    x.push_back((T)i);
  }
  typename vector<T>::iterator first = x.begin();
  typename vector<T>::iterator last  = x.end();

  while(first!=last) {
     T tmp = *first++;
     printf("%d ",(int)(tmp));
  }
  printf("%s\n",typeid(x).name());
}

template<class T>
void lfunc(T dmy) {
  list<T> x;
  for(int i=0;i<5;i++) {
    x.push_back((T)i);
  }
  typename list<T>::iterator first = x.begin();
  typename list<T>::iterator last  = x.end();

  while(first!=last) {
     printf("%d ",(int)*first++);
  }
  printf("%s\n",typeid(x).name());
}

main() {
  func(double());
  func(float());
  func(int());
  func(char());
  func(short());
  func(long());
#if 1
  func((unsigned int)0);
  func((unsigned char)0);
  func((unsigned short)0);
  func((unsigned long)0);
#endif

  lfunc(double());
  lfunc(int());
  lfunc(long());
  lfunc(float());
#if 0
  // Need to compile list containers
  lfunc(char());
  lfunc(short());
#endif
}

