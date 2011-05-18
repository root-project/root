/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// simple template test
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

// Simple template test
template<class T> class A {
 public:
    T a;
    T *p;
    A *next;
    A() { 
      a = 111;
      next=NULL; 
    }

    void disp() {
      cout << "a=" << a << "\n";
    }
};

A<int> a;

void test1()
{
  cout << "test1\n";
  A<double> b;
#ifdef NEVER
  printf("sizeof(a)=%d\n",sizeof(a));
  printf("sizeof(b)=%d\n",sizeof(b));
  printf("sizeof(A<int>)=%d\n",sizeof(A<int>));
  printf("sizeof(A<char>)=%d\n",sizeof(A<char>));
  printf("offsetof(A<int>,next)=%d\n",offsetof(A<int>,next));
#endif

  a.disp();
  b.disp();
}

// template with multiple argument and constant argument
template<class T,class E,int SZ> class B {
 public:
  T t[SZ];
  E e[SZ];
  B(T os=0) {
    int i;
    for(i=0;i<SZ;i++) {
      t[i] = i+os;
      e[i] = -i+os;
    }
  };
  void disp() {
    int i;
    for(i=0;i<SZ;i++) {
      cout << t[i] << " " << e[i] << "\n";
    }
  }
};

void test2()
{
  cout << "test2\n";
  B<int,double,5> b;
#ifdef NEVER
  printf("sizeof(b)=%d\n",sizeof(b));
#endif
  b.disp();
#ifdef NEVER
  printf("sizeof(B<int,double,5>)=%d\n",sizeof(B<int,double,5>));
  printf("sizeof(B<int,int,10>)=%d\n",sizeof(B<int,int,10>));
#endif
  B<int,int,10> c(10);
#ifdef NEVER
  printf("sizeof(c)=%d\n",sizeof(c));
  printf("offsetof(B<int,double,5>,e)=%d\n",offsetof(B<int,double,5>,e));
#endif
  c.disp();

  cout << "casting test\n";
  void *p;
  p = &c;
  ((B<int,int,10>*)p)->disp();
}

// Inheritance and template
class C : B<short,short,4> {
  int c;
 public:
  C(short os=0);
  void display();
};

C::C(short os) : B<short,short,4>(os) {
  c=os;
}
void C::display() {
  disp();
}

#ifdef NEVER
void sub(int i)
{
  cout << "sub(" << i << ")\n";
  B<int,int,i> d(50*i);
  d.disp();
}
#endif

void test3()
{
  cout << "test3\n";
  C c(20);
  c.display();

  B<int,int,5*2> d=B<int,int,10>(100);
  // B<int,int,5*2> d(100);
  d.disp();

#ifdef NEVER
  int i;
  for(i=2;i<5;i++) sub(i);
#endif
}


// typedef test
typedef B<double,double,3> bdouble3;

void test4()
{
  cout << "test4\n";
  bdouble3 e(1.5);
  e.disp();
}

void test5()
{
  cout << "test5\n";
  B<float,float,4> *p=new B<float,float,4>(0.2);
  p->disp();
  delete p;
}

int main()
{
  test1();
  test2();
  test3();
  test4();
  //test5();  // not supported yet
  return 0;
}
