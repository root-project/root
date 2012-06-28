/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// file arrayfunc.C
#include <stdio.h>
#include <iostream>

typedef float array_t[4];

float myarray[4][4];

//#if defined(__MAKECINT__)
//#else
const array_t& arrayret(int i) {

   for(int i=0; i<4; i++)
      for(int j=0; j<4; j++)
         myarray[i][j] = (i+1)*10+(j+1);

   void *v = &(myarray[0][0]);

   array_t *arr = (array_t*)v;
   // array_t &ret( arr[1] );

   return arr[1];

   // Interpreting arrayret causes segv when destrying arr. float *arr[]
}
//#endif

void arrayfunc() {

   float f = arrayret(1)[1];

   fprintf(stdout,"the printed value should equal to 22: %f\n",f);
   std::cout << "should also be 22: " << arrayret(1)[1] << std::endl;
}

