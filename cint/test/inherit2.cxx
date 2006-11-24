/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// class inheritance test 2
//
// same member function name
// inheritance from header information and later bind the body
//
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif


class A {
       protected:
	 int a1;
       public:
	 A(int i);
	 int A1(void);
	 void disp(void) {  cout << "a1=" << a1 << "\n"; }
 };

 class B : public A {
	 int b1;
       public:
	 int B1(void);
	 B(int i);
	 void disp(void) { cout <<"a1="<<a1<<" b1="<<b1<<"\n"; }
 };

 A::A(int i)
 {
	 a1=i;
 }

 int A::A1(void)
 {
	 return(a1);
 }

 B::B(int i)
 : A(i+1)
 {
	 b1=i;
 }

 int B::B1(void)
 {
	 return(b1);
 }

 int main()
 {
	 A Aobj=A(15);
	 B Bobj=B(3);

	 Aobj.disp();
	 Bobj.disp();

	return 0;
}
