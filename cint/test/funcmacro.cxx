/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#define FUNCMACRO1(a,b)   a=b;                       \
                         printf("a=%d b=%d\n",a,b);
#define FUNCMACRO2(a)     a++;                       \
                         printf("a=%d\n",a);
#define ASSIGN(cast,object)                          \
                    *(cast *)p = object;

#define PRINT(fmt,cast)                              \
                    printf(fmt,*(cast *)p);

#define DOTEST(fmt,cast,object)                      \
          ASSIGN(cast,object)                        \
          PRINT(fmt,cast)      NULL;

int main()
{
	int i,j;
	void *p;
	
	printf("simple test\n");
	for(j=0;j<3;j++) {
		FUNCMACRO1(i,1)
		FUNCMACRO2(i)
		FUNCMACRO1(i,2)
		FUNCMACRO2(i)
		FUNCMACRO2(i)
	}

	
	printf("casting test\n");
	p = malloc(20);

	ASSIGN(int,i)
	PRINT("*(int*)p=%d\n",int)

	DOTEST("*(double*)p=%g\n",double,3.14)

	DOTEST("*(unsigned char*)p=%d\n",unsigned char,255)

	free(p);

	return(0);
}
       
