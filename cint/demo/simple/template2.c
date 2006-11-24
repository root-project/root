/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// cint test of global function template
//
#include <stdio.h>

///////////////////////////////////////////////////////////////////////
// test3 
///////////////////////////////////////////////////////////////////////
template<class T> class B {
 public:
  B(T in=0) { b = in; }
  T b;
};

template<class T> int compare3(B<T> a,B<T> b) {
  return(a.b==b.b);
}

void test3()
{
  int i;
  B<float> b1,b2(0.02);
  for(i=0;i<5;i++) {
    b1.b = i/100.0;
    printf("%d=compare3(%g,%g)\n",compare3(b1,b2),b1.b,b2.b);
  }
}

///////////////////////////////////////////////////////////////////////
// test1 
///////////////////////////////////////////////////////////////////////
template<class T> int compare(T a,T b) {
  if(a==b) return(1);
  else     return(0);
}

class A {
 public:
  A(int in=0) { a = in ; }
  int a;
};

int operator==(A a,A b) {
  return(a.a==b.a);
}

void test1()
{
  int i;
  int i1,i2=2;
  double d1,d2=0.2;
  A a1,a2(20);

  for(i=0;i<5;i++) {
    i1=i;
    d1 = i/10.0;
    a1.a=i*10;
    //printf("%d=compare(%d,%d)  %d=compare(%g,%g)\n"
	   //,compare(i1,i2),i1,i2 ,compare(d1,d2),d1,d2);
    printf("%d=compare(%d,%d)  %d=compare(%g,%g) %d=compare(%d,%d)\n"
	   ,compare(i1,i2),i1,i2 ,compare(d1,d2),d1,d2
	   ,compare(a1,a2),a1.a,a2.a);
  }
}

///////////////////////////////////////////////////////////////////////
// test2 
///////////////////////////////////////////////////////////////////////
template<class T,class E> int compare2(T a,E b) {
  if(a==b) return(1);
  else     return(0);
}

void test2()
{
  int i;
  A a(2);
  for(i=0;i<5;i++) {
    printf("%d=compare2(%d,%d)\n",compare2(i,a),i,a.a);
  }
  for(i=0;i<5;i++) {
    printf("%d=compare2(%d,%d)\n",compare2(a,i),a.a,i);
  }
}

///////////////////////////////////////////////////////////////////////
// test4 
///////////////////////////////////////////////////////////////////////
#ifdef __CINT__
template<class T,template<class U> class E> int cmp(E<T> a,E<T> b) {
  return(a.b==b.b);
}
#endif

void test4()
{
#ifdef __CINT__
  int i;
  B<short> b1,b2(200);
  for(i=0;i<5;i++) {
    b1.b = i*100;
    printf("%d=cmp(%d,%d)\n",cmp(b1,b2),b1.b,b2.b);
  }
#endif
}

///////////////////////////////////////////////////////////////////////
// test5 
///////////////////////////////////////////////////////////////////////
template<size_t SIZE,class T> class ary {
 public:
  ary(int set=SIZE) ;
  T a[SIZE];
  size_t size;
};

template<size_t SIZE,class T> ary<SIZE,T>::ary(int set) {
  size = set;
}

const size_t sz=5;

void test5()
{
  ary<sz,B<int> > a;
  printf("a.size=%d\n",a.size);
}

///////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////
main()
{
  test5();
  test4();
  test3();
  test2();
  test1();
}
