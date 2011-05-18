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


void test2()  // Array<double> 
{
  printf("This example contains illegal C++ template function call\n");
  printf("Array<double> test\n");
  darray x(0.0,9.0,10) , y[5];
  int n;
  for(n=0;n<5;n++) {
    if(n%2) y[n] = x*n*(-1);
    else    y[n] = x*n*2;
  }

  // exit();
  int i;
  for(n=0;n<5;n++) {
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
}
