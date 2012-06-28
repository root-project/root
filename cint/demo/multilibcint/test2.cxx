/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Test program that is run by multi-thread libcint

int main(int argc,char** argv) {
  int n;
  printf("test2 started\n");
  if(argc>1) n = atoi(argv[1]);
  else       n = 10;
  for(int i=0;i<n;i++) {
    for(int j=0;j<1000000;j++); 
    printf("test i=%d\n",i);
  }
  printf("test2 finished\n");
  return(0);
}


