//
//  AnalyticalIntegrals.cxx
//  
//
//  Created by Aurélie Flandi on 09.09.14.
//
//

#include <stdio.h>
#include "TROOT.h"
#include "TMath.h"
#include "AnalyticalIntegrals.h"
#include "Math/DistFuncMathCore.h" //for cdf


using namespace std;

Double_t AnalyticalIntegral(TF1 *f, Double_t a, Double_t b)
{

   Double_t xmin = a;
   Double_t xmax = b;
   Int_t    num  = f->GetNumber();
   Double_t *p   = f->GetParameters();
   Double_t result = 0.;
   
   if      (num == 200)//expo: exp(p0+p1*x)
   {
      result = (exp(p[0])/p[1])*(exp(-p[1]*xmin)-exp(-p[1]*xmax));
   }
   else if (num == 100)//gaus: [0]*exp(-0.5*((x-[1])/[2])^2))
   {
      result =  p[0]*sqrt(2*M_PI)*p[2]*(ROOT::Math::gaussian_cdf(xmax, p[2], p[1])- ROOT::Math::gaussian_cdf(xmin, p[2], p[1]));//
   }
   /*else if (num == ?)gaussn
    {
    result =  ROOT::Math::gaussian_cdf(xmax, p[2], p[1])- ROOT::Math::gaussian_cdf(xmin, p[2], p[1]);
    }
    */
   else if (num == 400)//landau: root::math::landau(x,mpv=0,sigma=1,bool norm=false)
   {
      result = p[1]*(ROOT::Math::landau_cdf(xmax,p[1],p[2]) - ROOT::Math::landau_cdf(xmin,p[1],p[2]));
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
