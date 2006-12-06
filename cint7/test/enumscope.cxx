/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// testing 
//   local enum used in default paraneter
//   macro function used in initialization
//

#include <stdio.h>

#define BIT(n)       (1 << (n))

int a3=BIT(3);
int a2=BIT(2);

enum bits { b1=BIT(1) , b2=BIT(2) };

class A {
      public:
	A() { }
	enum fruit { apple, orange, pineapple };

	void f(int a = apple) {
		printf("%d\n",a);
	}
	void g(int a = 0) {
		printf("%d %d\n",a,pineapple);
	}
};

int main()
{
	A a;
	for(int i=0;i<4;i++) {
		a.f();
		a.f(A::orange);
		a.g();
	}
	printf("%d %d %d %d\n",a3,a2,b1,b2);
	return 0;
}

