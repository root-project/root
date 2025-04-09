#include <stdio.h>
void t3()
{
double epsilon = 10e-7;
float f1 = 3.3333333;
float f2 = 3.3333334;
fprintf(stderr,"%f %lf %le\n",f1,f2,f1-f2);
if ( f1==f2 ) {
  fprintf(stderr,"strictly equal\n");
}
if ( fabs((f1-f2)/f1)<epsilon ) {
  fprintf(stderr,"equal\n");
}
}

