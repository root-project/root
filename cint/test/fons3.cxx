/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
typedef double Double_t;
#include <stdio.h>

// file: test.C
//____________________________________________________________________________ 
//  tfun1() is used in TF2 constructor and called by TF2.Draw()
//

Double_t tfun1(Double_t* x, Double_t* par)
{
  Double_t res;

  // Uncomment one of the following two lines.  The second one
  // will cause TF2::Draw() to never execute this function.
  Double_t xe[4];                           // ***** THIS WORKS ON SGI *****>
   //Double_t xe[4] = {0.0, 0.0, 0.0, 0.0};  // ***** THIS DOESN'T *****

  xe[0] = x[0] + 0.845;
  xe[1] = x[0] * x[1] * 0.246;
  xe[2] = 0.941;

  res = xe[0]+xe[1]+xe[2];

  printf("Result = %f\n", res);
  return res;
}

Double_t xf[20];
Double_t tfun2(Double_t* x, Double_t* par)
{
  Double_t res;

  // Uncomment one of the following two lines.  The second one
  // will cause TF2::Draw() to never execute this function.
   //Double_t xe[4] = {0.0, 0.0, 0.0, 0.0};  // ***** THIS DOESN'T *****

  xf[0] = x[0] + 0.845;
  xf[19] = x[0] * x[1] * 0.246;
  xf[2] = 0.941;

  res = xf[0]+xf[19]+xf[2];

  printf("Result = %f\n", res);
  return res;
}

Double_t tfun3(Double_t* x, Double_t* par)
{
  Double_t res;

  // Uncomment one of the following two lines.  The second one
  // will cause TF2::Draw() to never execute this function.
   Double_t xe[4] = {0.0, 0.0, 0.0, 0.0};  // ***** THIS DOESN'T *****

  xe[0] = x[0] + 0.845;
  xe[1] = x[0] * x[1] * 0.246;
  xe[2] = 0.941;

  res = xe[0]+xe[1]+xe[2];

  printf("Result = %f\n", res);
  return res;
}

int main() {
  Double_t x[] = { 1, 2, 3, 4, 5, 6, 7} ;
  Double_t par[3] ;
  int i;
  printf("\n");
  for(i=0;i<3;i++) tfun1(x+i,par);
  for(i=0;i<3;i++) tfun2(x+i,par);
  for(i=0;i<3;i++) tfun3(x+i,par);
  return 0;
}

