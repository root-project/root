/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* #! /usr/local/bin/cint  */
#include <stdio.h>
main(argc,argv)
int argc;
char *argv[];
{
	int i;
	printf("argc=%d\n",argc);
	for(i=0;i<argc;i++) {
		printf("argv[%d]=\"%s\" , sum=%d\n"
		       ,i,argv[i],sumchar(argv[i]));
	}
}

int sumchar(char *string)
//int sumchar(string)
//char *string;
{
	int sum=0,i=0;
	while(string[i]!='\0') {
		sum += string[i++];
	}
	return(sum);
}
