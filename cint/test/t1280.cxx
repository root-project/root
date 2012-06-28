/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
typedef double (*myfunc)(double*, double*);

class TF1 {
public:
   TF1(const char*,myfunc,int,int,int) {};
};

double a0 (double* xa, double* par);
double a1 (double* xa, double* par);
double a2 (double* xa, double* par);
double a3 (double* xa, double* par);
double a4 (double* xa, double* par);

double f0 (double* xa, double* par)
{
  return 0;
}

double f1 (double* xa, double* par)
{
  return 1;
}

double f2abc (double* xa, double* par)
{
  return 2;
}

double u1 (double* xa, double* par)
{
  static TF1 sf1 ("tmp1", f1, 0, 100, 1);

  return 3;
}

// Commenting out either u2 or u3 will enable rootcint to parse u4
double u2 (double* xa, double* par)
{
  static TF1 sf2 ("tmp2", f2abc, 0, 100, 1);

  return 4;
}

double u3 (double* xa, double* par)
{
  static TF1 sf1 ("tmp3", f0, 0, 100, 1);

  return 3;
}

double u4 (double* xa, double* par)
{
  static TF1 sf2 ("tmp4", f1, 0, 100, 1);

  return 4;
}

#include <stdio.h>

int main(int,char**) 
{
   // nothing to do
   printf("running successfully\n");
   return 0;
}
