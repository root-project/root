/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// baseconv0.c
//
//
#include <stdio.h>

class A {
      public:
	int i;
	void Copy(A& a) { i=a.i; }
	void copy(int a) { i=a; }
	void f(A& a,int b) {  i = a.i+b; }
	A(int a) { i=a; }
	void disp() { printf("A::i=%d\t",i); }
};

class B {
      public:
	int i;
	void Copy(B& a) { i=a.i; }
	void copy(int a) { i=a; }
	void f(B& a,int b) {  i = a.i+b; }
	B(int a) { i=a; }
	void disp() { printf("B::i=%d\n",i); }
};

class D : public A, public B {
      public:
	int i;
	D(int a) : A(a+1) , B(a+2) { i=a; }
	void disp() { printf("D::i=%d\t",i); A::disp(); B::disp(); }
};

int main()
{
	A a(10);
	B b(20);
	D d(30),dd(40);

	// d.Copy(a);  // ambiguous A::Copy() and B::Copy(), OK in cint
	// d.Copy(b);  // ambiguous A::Copy() and B::Copy(), OK in cint

	d.disp();

	d.A::Copy(dd); d.disp();
	d.B::Copy(dd); d.disp();

	d.A::Copy(a);  d.disp(); 
	d.B::Copy(b);  d.disp();

	d.A::copy(1);  d.disp();
	d.B::copy(2);  d.disp();

	d.A::f(dd,3);   d.disp();
	d.B::f(dd,3);   d.disp();
	d.A::f(a,3);   d.disp();
	d.B::f(b,3);   d.disp();

	for(int i=0;i<5;i++) {
		d.A::Copy(dd); d.disp();
		d.B::Copy(dd); d.disp();
		
		d.A::Copy(a);  d.disp(); 
		d.B::Copy(b);  d.disp();

		d.A::copy(i);  d.disp();
		d.B::copy(i+1);  d.disp();

		d.A::f(dd,i+4);   d.disp();
		d.B::f(dd,i+5);   d.disp();
		d.A::f(a,i+6);   d.disp();
		d.B::f(b,i+7);   d.disp();
	}
	return 0;
}
