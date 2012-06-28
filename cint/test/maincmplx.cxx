/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include "complex1.h"
#include <math.h>


void test1()
{
	complex a(1,2),b(3,4),c;

	c = a+b;

	printf("%g %g\n",c.re,c.im);
}

void test2()
{
	int i;
	complex a[10],b[10],c[10];

	for(i=0;i<10;i++) {
		a[i].re=i;
		a[i].im=i*2;
		b[i].re=i*3;
		b[i].im=i*5;
	}
	
	for(i=0;i<10;i++) {
		c[i]=b[i]-a[i];
	}
	for(i=0;i<10;i++) {
		printf("c[%d] %g %g\n",i,c[i].re,c[i].im);
	}
}

void test3()
{
	int i;
	complex *p,c,*pc,C;
	pc=&C;

	for(i=0;i<5;i++) {
		p = new complex(i,i*2);
		c = c+(*p);
		*pc = (*pc)+(*p);
		delete p;
	}

	printf("c %g %g\n",c.re,c.im);
	printf("*pc %g %g\n",pc->re,pc->im);
}

void test4()
{
	complex i;
	for(i=complex(0);i<=5;i=1+i) {
		printf("%g %g : ",i.re,i.im);
	}
	printf("%g %g\n",i.re,i.im);
}

void test5()
{
  int i;
  complex a(5,6),b(7,8),c;
  
  for(i=0;i<4;i++) {
    c(i-1,i)=a+b(i,i+1);
    printf("b.re=%g b.im=%g c.re=%g c.im=%g\n"
	   ,b.re,b.im,c.re,c.im);
  }
  
  for(i=0;i<4;i++) {
    c[i] = (i+1)*20;
    printf("c[0]=%g c[1]=%g\n",c[0],c[1]);
  }
  printf("c[0]=%g c[1]=%g\n",c[0],c[1]);

}

void test6()
{
	enum fruits myfavorite;

	myfavorite = apple;

	printf("%d %d %d %d\n",orange,apple,others,myfavorite);
}

int main()
{
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
	return 0;
}
