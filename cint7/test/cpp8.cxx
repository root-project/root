/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**********************************************************************
* cpp8.cxx
*
* New delete operator and array object construction/destruction test
**********************************************************************/
#include <stdlib.h>
#include <stdio.h>

int i;

class complex {
	double re,im;
public:
	complex(double a) { re=a; im=0; }
	complex(void) { re=i++; im=i++;}
	void print() { printf("%g %g\n",re,im);}
	~complex(void) { this->print();}
};

void test3()
{
	int j,k=5;
	complex c[3];
	complex *d,*e,*f;

	printf("c\n");
	for(j=0;j<3;j++) c[j].print();

	printf("d\n");
	d = new complex[1+2+1];
	for(j=0;j<4;j++) d[j].print();
	delete[] d;

	printf("e and d\n");
	e = new complex[k*2];
	d = new complex;
	delete [] e;
	delete d;

	printf("e and d and f\n");
	e = new complex[k-1];
	d=new complex[k];
	f = new complex [2];
	delete[] e;
	delete [] d;

	printf("e and d\n");
	e = new complex[k-2];
	d=new complex[k+1];
	delete [] d;
	delete[] e;
	delete[] f;

	printf("end\n");
}



void test2()
{
	double *px;
	int i,size=20;

	px = new double[size];
	for(i=0;i<size;i++) px[i]=i*1.5;

	printf("px[]=");
	for(i=0;i<size;i++) printf("%g ",px[i]);
	printf("\n");

	delete[] px;
}

void test1()
{
	int *pi;
	unsigned short *pus;

	pi = new int;
	pus =new unsigned short ;

	*pi=1;
	*pus=0xfffff;
	printf("*pi=%d *pus=%d\n",*pi,*pus);

	delete pus;
	delete pi;

}

int main()
{
	test1();
	test2();
	test3();
	return 0;
}
