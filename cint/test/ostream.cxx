/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*********************************************************************
* ostream.c
*
* output stream test
*********************************************************************/
#ifdef __hpux
#include <iostream.h>   
#else
#include <iostream>   
using namespace std;
#endif
// important

class complex {
 public:
	double re,im;
	complex(double a=0.0, double b=0.0) { re=a; im=b;}
};

ostream& operator <<(ostream& ios,complex a)
{
	ios << '(' << a.re << ',' << a.im << ')' ;
	return(ios);
}

int main()
{
	char c='a';
	unsigned short b=255;
	const char *pc="hijklmn";
	//unsigned char *pb=(unsigned char*)255;

	short s=0xffff;
	unsigned short r=0xffff;
	//short *ps=(short*)0xffff;
	//unsigned short *pr=(unsigned short*)0xffff;

	int I=3229;
	unsigned int h=0xffffffff;
	//int *pI=(int*)3229;
	//unsigned int *ph=(unsigned int*)0xffffffff;

	long l=-1234567;
	unsigned long k=15;
	//long *pl=(long*)-1234567;
	//unsigned long *pk=(unsigned long*)15;

	double pi=3.141592;
	float f=3.14;
	double d=-1e-15;
	//float *pf=(float*)314;
	//double *pd=(double*)-115;

	complex C(1.5,2.5);

	cout << "I=" << I << "  pi=" << pi << '\n';
	cout << "(complex)C="<<C << '\n';

	cout<<"c="<<c<<'\n';
	cout<<"b="<<b<<'\n';

	cout << "s=" << s << '\n';
	cout << "r=" << r << '\n';

	cout<<"I="<<I<<'\n';
	cout<<"h="<<h<<'\n';

	cout<<"l="<<l<<'\n';
	cout<<"k="<<k<<'\n';

	cout<<"f="<<f<<'\n';
	cout<<"d="<<d<<'\n';

	cout<<"pc="<<pc<<'\n';

#if 0
	cout<<"pb="<<pb<<'\n';

	cout << "ps=" << ps << '\n';
	cout << "pr=" << pr << '\n';

	cout<<"pI="<<pI<<'\n';
	cout<<"ph="<<ph<<'\n';

	cout<<"pl="<<pl<<'\n';
	cout<<"pk="<<pk<<'\n';

	cout<<"pf="<<pf<<'\n';
	cout<<"pd="<<pd<<'\n';
#endif

	return 0;
}

