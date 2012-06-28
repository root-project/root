/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// demo/exception/ehdemo.cxx
// This source has to be interpreted

#include <stdio.h>

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1046.h"
#endif


////////////////////////////////////////////////////////////////////////
// You can use simple try, catch block in an interpreted code. However,
// that portion can not be bytecode optimized. 
////////////////////////////////////////////////////////////////////////

void test1() {
  try {
	int iRows = 10000;
	int iCols = 10000;

	IntVar iv0(2,2);
	
	// When the braces are removed,
	// everything is OK
	//if (iRows > 1) {IntVar iv1(iRows, iCols);}
	IntVar iv1(iRows, iCols);
	

	printf("after exception has been thrown.");
	IntVar iv2(2,2);
	
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
	
  }
 
}

void test2() {
  try {
	int iRows = 10000;
	int iCols = 10000;

	IntVar iv0(2,2);
	
	// When the braces are removed,
	// everything is OK
	if (iRows > 1) {IntVar iv1(iRows, iCols);}
	//IntVar iv1(iRows, iCols);
	

	printf("after exception has been thrown.");
	IntVar iv2(2,2);
	
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
	
  }
 
}

void test3() {
  try {
	int iRows = 10000;
	int iCols = 10000;

	IntVar iv0(2,2);
	
	// When the braces are removed,
	// everything is OK
	for(;;) {IntVar iv1(iRows, iCols);}
	//IntVar iv1(iRows, iCols);
	

	printf("after exception has been thrown.");
	IntVar iv2(2,2);
	
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
	
  }
 
}

void test4() {
  try {
	int iRows = 10000;
	int iCols = 10000;

	IntVar iv0(2,2);
	
	// When the braces are removed,
	// everything is OK
	while(1) {IntVar iv1(iRows, iCols);}
	//IntVar iv1(iRows, iCols);
	

	printf("after exception has been thrown.");
	IntVar iv2(2,2);
	
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
	
  }
 
}


void test5() {
  try {
	int iRows = 10000;
	int iCols = 10000;

	IntVar iv0(2,2);
	
	// When the braces are removed,
	// everything is OK
	do {IntVar iv1(iRows, iCols);} while(1);
	//IntVar iv1(iRows, iCols);
	

	printf("after exception has been thrown.");
	IntVar iv2(2,2);
	
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
	
  }
 
}

void test6() {
  try {
	int iRows = 10000;
	int iCols = 10000;

	IntVar iv0(2,2);
	
	// When the braces are removed,
	// everything is OK
	switch(iRows) {
	case 0:
	case 1:
	  break;
	default:
	  IntVar iv1(iRows, iCols);
	  break;
	} 
	//IntVar iv1(iRows, iCols);
	

	printf("after exception has been thrown.");
	IntVar iv2(2,2);
	
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
	
  }
 
}

void test7() {
  try {
	int iRows = 10000;
	int iCols = 10000;

	IntVar iv0(2,2);
	
	// When the braces are removed,
	// everything is OK
	switch(iRows) {
	case 0:
	case 1:
	  break;
	default:
	  if(1) {
	    for(;;) {
	      IntVar iv1(iRows, iCols);
	    }
	  }
	  break;
	} 
	//IntVar iv1(iRows, iCols);
	

	printf("after exception has been thrown.");
	IntVar iv2(2,2);
	
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
	
  }
 
}


////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
int main() {
  printf("------- START -------\n");
  // Calling compiled function that throws exception with interpreted
  // try, catch block

  printf("test1 ");
  test1();
  printf("test2 ");
  test2();
  printf("test3 ");
  test3();
  printf("test4 ");
  test4();
  printf("test5 ");
  test5();
  printf("test6 ");
  test6();
  printf("test7 ");
  test7();

  printf("-------- END --------\n");
  return 0;
}
