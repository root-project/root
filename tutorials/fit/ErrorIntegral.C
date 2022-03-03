/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// Estimate the error in the integral of a fitted function
/// taking into account the errors in the parameters resulting from the fit.
/// The error is estimated also using the correlations values obtained from
/// the fit
///
/// run the macro doing:
///
/// ~~~{.cpp}
///  .x ErrorIntegral.C
/// ~~~
///
/// After having computed the integral and its error using the integral and the integral
/// error using the generic functions TF1::Integral and TF1::IntegralError, we compute
/// the integrals and its error analytically using the fact that the fitting function is
/// \f$ f(x) = p[1]* sin(p[0]*x) \f$.
///
/// Therefore we have:
///  - integral  in [0,1] : `ic = p[1]* (1-std::cos(p[0]) )/p[0]`
///  - derivative of integral with respect to  p0:
///    `c0c = p[1] * (std::cos(p[0]) + p[0]*std::sin(p[0]) -1.)/p[0]/p[0]`
///  - derivative of integral with respect to p1:
///   `c1c = (1-std::cos(p[0]) )/p[0]`
///
/// and then we can compute the integral error using error propagation and the covariance
/// matrix for the parameters p obtained from the fit.
///
/// integral error :    `sic = std::sqrt( c0c*c0c * covMatrix(0,0) + c1c*c1c * covMatrix(1,1) + 2.* c0c*c1c * covMatrix(0,1))`
///
/// Note that, if possible, one should fit directly the function integral, which are the
/// number of events of the different components (e.g. signal and background).
/// In this way one obtains a better and more correct estimate of the integrals uncertainties,
/// since they are obtained directly from the fit without using the approximation of error propagation.
/// This is possible in ROOT. when using the TF1NormSum class, see the tutorial fitNormSum.C
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Lorenzo Moneta

#include "TF1.h"
#include "TH1D.h"
#include "TFitResult.h"
#include "TMath.h"
#include <assert.h>
#include <iostream>
#include <cmath>

TF1 * fitFunc;  // fit function pointer

const int NPAR = 2; // number of function parameters;

//____________________________________________________________________
double f(double * x, double * p) {
   // function used to fit the data
   return p[1]*TMath::Sin( p[0] * x[0] );
}

//____________________________________________________________________
void ErrorIntegral() {
   fitFunc = new TF1("f",f,0,1,NPAR);
   TH1D * h1     = new TH1D("h1","h1",50,0,1);

   double  par[NPAR] = { 3.14, 1.};
   fitFunc->SetParameters(par);

   h1->FillRandom("f",1000); // fill histogram sampling fitFunc
   fitFunc->SetParameter(0,3.);  // vary a little the parameters
   auto fitResult = h1->Fit(fitFunc,"S");             // fit the histogram and get fit result pointer

   h1->Draw();

   /* calculate the integral*/
   double integral = fitFunc->Integral(0,1);

   auto covMatrix = fitResult->GetCovarianceMatrix();
   std::cout << "Covariance matrix from the fit ";
   covMatrix.Print();

   // need to pass covariance matrix to fit result.
   // Parameters values are are stored inside the function but we can also retrieve from TFitResult
   double sigma_integral = fitFunc->IntegralError(0,1, fitResult->GetParams() , covMatrix.GetMatrixArray());

   std::cout << "Integral = " << integral << " +/- " << sigma_integral
             << std::endl;

   // estimated integral  and error analytically

   double * p = fitFunc->GetParameters();
   double ic  = p[1]* (1-std::cos(p[0]) )/p[0];
   double c0c = p[1] * (std::cos(p[0]) + p[0]*std::sin(p[0]) -1.)/p[0]/p[0];
   double c1c = (1-std::cos(p[0]) )/p[0];

   // estimated error with correlations
   double sic = std::sqrt( c0c*c0c * covMatrix(0,0) + c1c*c1c * covMatrix(1,1)
      + 2.* c0c*c1c * covMatrix(0,1));

   if ( std::fabs(sigma_integral-sic) > 1.E-6*sic )
      std::cout << " ERROR: test failed : different analytical  integral : "
                << ic << " +/- " << sic << std::endl;
}
