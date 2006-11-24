/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

#define NUM 10
void test3()
{
	int i=0;
	float ary[NUM];
	while(i<NUM){  
		ary[i] = i;
		ary[i] = -i;
		ary[i] = i*2;
		++i;
	} 
}

void test2()
{
	int i=0,j=0;
	while(i<10 && j!=4) {
		printf("i=%d j=%d\n",i,j);
		i++; j++;
	}
	printf("i=%d j=%d\n",i,j);

	while(i>0 || j==0) {
		printf("i=%d j=%d\n",i,j);
		i--;j--;
	}
	printf("i=%d j=%d\n",i,j);
}

void test1()
{
	int i;
	for(i=0;i<4;i++) {
		if(((i==1)&&(2==2))||(2==3)) {
			printf("true\n");
		}
		else {
			printf("false\n");
		}
		if(((i==2)&&(2==2))||(2==3)) {
			printf("true\n");
		}
		else {
			printf("false\n");
		}
		if(((i==2)&&(2==2))||(3==3)) {
			printf("true\n");
		}
		else {
			printf("false\n");
		}
	}
}

int main()
{
	test3();
	test1();
	test2();
	return 0;
}
