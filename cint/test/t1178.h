/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
template<class T> class vector {
 public:
  typedef T* iterator;
};

typedef double FPoint;
typedef vector <FPoint *> PointVector;
typedef vector <FPoint *> :: iterator PointVectorIterator;

class CTest
{
 public:
  CTest( void ) { printf("CTest::CTest()\n"); return; };
  ~CTest() { printf("CTest::~CTest()\n"); return; };
  
  void Execute ( PointVectorIterator &iter ) {
    printf("CTest::Execute(PointVectorIterator&)\n");
  }
  void f(double**& x) {
    printf("CTest::f(double**&)\n");
  }
  void g(vector<double*>::iterator& x) {
    printf("CTest::g(vector<double*>&)\n");
  }
  void h(vector<FPoint*>::iterator& x) {
    printf("CTest::h(vector<FPoint*>&)\n");
  }
};


