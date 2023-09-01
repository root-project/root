// @(#)root/hist:$Id$
// Authors: L. Moneta, A. Flandi 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  ROOT  Team, CERN/PH-SFT                        *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
//
//  AnalyticalIntegrals.cxx
//
//
//  Created by Aurélie Flandi on 09.09.14.
//
//

#include "AnalyticalIntegrals.h"

#include "TROOT.h"
#include "TF1.h"
#include "TFormula.h"
#include "TMath.h"
#include "Math/DistFuncMathCore.h" //for cdf

#include <stdio.h>

using namespace std;

Double_t AnalyticalIntegral(TF1 *f, Double_t a, Double_t b)
{

   Double_t xmin = a;
   Double_t xmax = b;
   Int_t    num  = f->GetNumber();
   Double_t *p   = f->GetParameters();
   Double_t result = 0.;

   TFormula * formula = f->GetFormula();
   if (!formula) {
      Error("TF1::AnalyticalIntegral","Invalid formula number - return a NaN");
      return TMath::QuietNaN();
   }

   if   (num == 200)//expo: exp(p0+p1*x)
   {
      result = ( exp(p[0]+p[1]*xmax) - exp(p[0]+p[1]*xmin))/p[1];
   }
   else if (num == 100)//gaus: [0]*exp(-0.5*((x-[1])/[2])^2))
   {
      double amp   = p[0];
      double mean  = p[1];
      double sigma = p[2];
      if (formula->TestBit(TFormula::kNormalized))
         result = amp * (ROOT::Math::gaussian_cdf(xmax, sigma, mean) - ROOT::Math::gaussian_cdf(xmin, sigma, mean));
      else
         result = amp * sqrt(2 * TMath::Pi()) * sigma *
                  (ROOT::Math::gaussian_cdf(xmax, sigma, mean) - ROOT::Math::gaussian_cdf(xmin, sigma, mean)); //
   }
   else if (num == 400)//landau: root::math::landau(x,mpv=0,sigma=1,bool norm=false)
   {

      double amp   = p[0];
      double mean  = p[1];
      double sigma = p[2];
      //printf("computing integral for landau in [%f,%f] for m=%f s = %f \n",xmin,xmax,mean,sigma);
      if (formula->TestBit(TFormula::kNormalized) )
         result = amp*(ROOT::Math::landau_cdf(xmax,sigma,mean) - ROOT::Math::landau_cdf(xmin,sigma,mean));
      else
         result = amp*sigma*(ROOT::Math::landau_cdf(xmax,sigma,mean) - ROOT::Math::landau_cdf(xmin,sigma,mean));
   }
   else if (num == 500) //crystal ball
   {
      double amp   = p[0];
      double mean  = p[1];
      double sigma = p[2];
      double alpha = p[3];
      double n     = p[4];

      //printf("computing integral for CB in [%f,%f] for m=%f s = %f alpha = %f n = %f\n",xmin,xmax,mean,sigma,alpha,n);
      if (alpha > 0)
         result = amp*( ROOT::Math::crystalball_integral(xmin,alpha,n,sigma,mean) -  ROOT::Math::crystalball_integral(xmax,alpha,n,sigma,mean) );
      else {
         result = amp*( ROOT::Math::crystalball_integral(xmax,alpha,n,sigma,mean) -  ROOT::Math::crystalball_integral(xmin,alpha,n,sigma,mean) );
      }
   }

   else if (num >= 300 && num < 400)//polN
   {
      Int_t n = num - 300;
      for (int i=0;i<n+1;i++)
      {
         result += p[i]/(i+1)*(std::pow(xmax,i+1)-std::pow(xmin,i+1));
      }
   }
   else
      result = TMath::QuietNaN();

   return result;
}
