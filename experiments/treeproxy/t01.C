
#include <stdio.h>

typedef float array[3];

const array* func() {
   static float f[2][3];
   f[1][2] = 5;
   f[2][1] = 7;
   void * vf = &(f[0][0]);
   array *q = ((array*)vf);
   return q;
}

void t01() {
   float f = func()[1][2];
   fprintf(stderr,"flaot is %f\n",f);
   f = func()[2][1];
   fprintf(stderr,"flaot is %f\n",f);
}

#ifdef __MAKECINT__
#pragma link C++ function func;
#endif

