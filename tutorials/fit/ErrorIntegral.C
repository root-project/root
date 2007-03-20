// Macro to estimate the error in the integral of a fitted function taking into account the 
// errors in the parameters resulting from the fit. 
// The error is estimated also using the correlations values obtained from the fit
//
// run the macro doing: 
//  .x ErrorIntegral.C
// Author: Lorenzo Moneta

#include "TF1.h"
#include "TH1D.h"
#include "TVirtualFitter.h"
#include "TMath.h"
#include <assert.h>
#include <iostream>

TF1 * fitFunc;  // fit function pointer (need to be global since it is used by the gradient functions )

const int NPAR = 2; // number of function parameters;

//____________________________________________________________________
double f(double * x, double * p) { 
   // function used to fit the data
   return p[1]*TMath::Sin( p[0] * x[0] ); 
}

//____________________________________________________________________
double df_dPar(double * x, double * p) { 
   // derivative of the function w.r..t parameters
   // use calculated derivatives from TF1::GradientPar

   double grad[NPAR]; 
   // p is used to specify for which parameter the derivative is computed 
   int ipar = int(p[0] ); 
   assert (ipar >=0 && ipar < NPAR );

   assert(fitFunc);
   fitFunc->GradientPar(x, grad);

   return grad[ipar]; 
}

//____________________________________________________________________
double IntegralError(int npar, double * c, double * errPar, double * covMatrix = 0) {   
// calculate the error on the integral given the parameter error and the integrals of 
// the gradient functions c[] 

   double err2 = 0; 
   for (int i = 0; i < npar; ++i) { 
      if (covMatrix == 0) { // assume error are uncorrelated
         err2 += c[i] * c[i] * errPar[i] * errPar[i]; 
      } else {
         double s = 0; 
         for (int j = 0; j < npar; ++j) {
            s += covMatrix[i*npar + j] * c[j]; 
         }
         err2 += c[i] * s; 
      }
   }

   return TMath::Sqrt(err2);
}

//____________________________________________________________________
void ErrorIntegral() { 
   fitFunc = new TF1("f",f,0,1,NPAR); 
   TH1D * h1     = new TH1D("h1","h1",50,0,1); 

   double  par[NPAR] = { 3.14, 1.}; 
   fitFunc->SetParameters(par);

   h1->FillRandom("f",1000); // fill histogram sampling fitFunc
   fitFunc->SetParameter(0,3.);  // vary a little the parameters
   h1->Fit(fitFunc);             // fit the histogram 

   h1->Draw();

   // calculate the integral 
   double integral = fitFunc->Integral(0,1);

   // calculate now the error (needs the derivatives of the function w..r.t the parameters)
   TF1 * deriv_par0 = new TF1("dfdp0",df_dPar,0,1,1);
   deriv_par0->SetParameter(0,0);

   TF1 * deriv_par1 = new TF1("dfdp1",df_dPar,0,1,1);
   deriv_par1->SetParameter(0,1.);

   double c[2]; 

   c[0] = deriv_par0->Integral(0,1); 
   c[1] = deriv_par1->Integral(0,1); 

   double * epar = fitFunc->GetParErrors();

   double sigma_integral = IntegralError(2,c,epar);

   std::cout << "Integral = " << integral << " +/- " << sigma_integral << std::endl;

   TVirtualFitter * fitter = TVirtualFitter::GetFitter();
   assert(fitter != 0);
   double * covMatrix = fitter->GetCovarianceMatrix(); 

   double sigma_integral_2 = IntegralError(2,c,epar,covMatrix);
   std::cout << "Error taking into account correlations:\t" << sigma_integral_2 << std::endl;
 
   double * p = fitFunc->GetParameters();
   double ic  = p[1]* (1-std::cos(p[0]) )/p[0];
   double c0c = p[1] * (std::cos(p[0]) + p[0]*std::sin(p[0]) -1.)/p[0]/p[0];
   double c1c = (1-std::cos(p[0]) )/p[0];

   double sic = std::sqrt( c0c*c0c * epar[0]*epar[0] + c1c*c1c * epar[1]*epar[1] ); 

   if ( std::fabs(sigma_integral-sic) > 1.E-6*sic ) 
      std::cout << " ERROR: test failed : different analytical  integral : " << ic << " +/- " << sic << std::endl;
}
