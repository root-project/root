/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <stdlib.h>
class A {
      public:
	int *p[10];
	void func() {
		int i;
		for(i=0;i<10;i++) p[i]=NULL;
		p[3]=(int*)malloc(4);
		for(i=0;i<5;i++) {
			if(p[i]) {
				free(p[i]);
				printf("p[%d] free\n",i);
			}
			printf("p[%d]=0\n",i);
		}
	}
};
int main()
{
  A obja;
  obja.func();
  return 0;
}
