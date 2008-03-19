/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// Array precompiled class simple test program
//

#define SIZE 12

void test2()  // Array<double> 
{
  printf("Array<double> test\n");
  darray x(1.0,10.0,10) , y[SIZE];
  int n;
  for(n=0;n<SIZE;n++) {
    if(n%2) {
      y[n] = x*x*x;
    }
    else {
      y[n] = x*x;
    }
  }

  int i;
  for(n=0;n<SIZE;n++) {
    printf("%d : ",n);
    for(i=0;i<10;i++) {
      printf("%g ",y[n][i]);
    }
    printf("\n");
  }
}

main()
{
  test2();
  // test3();
}
