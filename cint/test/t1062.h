/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// 1112, 1155, 1156, 1157, 1159

#if 1

//class TString;
#include <vector>
using namespace std;

#else

char const*& f();

template<class T> class vector { 
  vector() { }
  vector(const vector& x) { }
  vector& operator=(const vector& x) { }
  ~vector() { }
 public:
  //T& operator[](long n);
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  //void deallocate(pointer p) ;
  //const_pointer const_address(const_reference x) ; 
  reference operator[](const int& __n) const ;
  T& operator[](const short& __n) const ;
};

#endif

class A {
};

class test {
  vector<const char*> vv1;
  vector<char const*> vv2;
  vector<const int*> vi1;
  vector<int const*> vi2;
  vector<const double*> vd1;
  vector<double const*> vd2;
  vector<const A*> va1;
  //vector<A const*> va2; // fread.c G__isstoragekeyword()
};

