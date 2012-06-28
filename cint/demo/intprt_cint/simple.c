/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
main()
{
  int i;
  double d[5];
  
  printf("\nDon't mind about error messages above. This is very slow, be patient.\n");
  printf("Compiled cint interpretes cint itself and the interpreted cint interpretes\n");
  printf("this simple source code 'simple.c'.\n\n");
  
  for(i=0;i<5;i++) {
    d[i] = exp(i);
    printf("test OK d[%d]=%g\n",i,d[i]);
  }
  
  printf("See! Cint interpreted by cimpiled cint can interpret a simple C program.\n");
}
