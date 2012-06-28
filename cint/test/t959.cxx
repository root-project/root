/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 021203loopbug.txt

#include <stdio.h>
void bug() {
  double* data = new double[10];
#if 0
  fprintf(stdout,"Addr: %p\n",data);
#endif
  data[0]=0;
  fprintf(stdout,"data[0]: %lf\n",data[0]);
#if 1
  fprintf(stdout,"((void*)(&data[1])-(void*)(&data[0]): %lx\n"
	  ,(long)(&data[1])-(long)(&data[0]));
#else
  fprintf(stdout,"&(data[0]): %p\n",&(data[0]));
  fprintf(stdout,"&(data[1]): %p\n",&(data[1]));
#endif
  int j;
  for (j = 0 ; j < 3; j++) {
     fprintf(stdout,"Executing loop %d\n",j);
     data[j+1] = j+1;
  }
  for (j = 0 ; j < 3; j++) {
     fprintf(stdout,"Executing loop %d with val %lf\n",j,data[j+1]);
  }
  delete[] data;
}

int main() {
  bug();
  return 0;
}

