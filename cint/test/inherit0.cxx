/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// class inheritance test 0
//
//  multiple inheritance
//  access control of member variable , private, protected
//
#include <stdio.h>

class A {
      protected:
	int a1;
	double a2;
      public:
	int A1() { return(a1); }
	double A2() { return(a2); }
};

class C {
      protected:
	unsigned char c1;
	double c2;
      public:
	unsigned char C1() { return(c1); }
	double C2() { return(c2); }
};

class B : public A, public C {
private:
  short b1;
  float b2;
  short BB1() { return(b1); }
public:
  void setvalues(int i,double j,short k,float l,unsigned char m,double n) {
    a1=i; a2=j; c1=m; c2=n; // protected 
    b1=k; b2=l;
    BB1();
  }
  
  short B1() { return b1 ; }
  float B2() { return(b2); }
};

class D : public B {
private:
  int d1;
public:
  int D1() { return(d1); }
  void set(int i) { d1=i; 
  //a1;a2; c1;c2; // protected member,
  //b1;  // access error, private member of base
  //BB1(); // access error, private member of base
  }
};

int main()
{
	D obj;

	obj.setvalues(1,3.14,2,6.26,3,1.6e-19);

	//fprintf(stderr,"!!!Intentional error a1,b1,BB1() below\n");
	//a1;
	obj.set(4); // intentional error

	printf("%d %g %d %g %d %g %d\n"
	       ,obj.A1(),obj.A2()
	       ,obj.B1(),obj.B2()
	       ,obj.C1(),obj.C2()
	       ,obj.D1()
	       );

	return 0;
}
