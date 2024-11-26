/// \file
/// \ingroup tutorial_fit
/// \notebook -nodraw
/// example of fitting a 3D function
/// Typical multidimensional parametric regression where the predictor
/// depends on 3 variables
///
/// In the case of 1 or 2D one can use the TGraph classes
/// but since no TGraph3D class exists this tutorial provide
/// an example of fitting 3D points
///
/// \macro_output
/// \macro_code
///
/// \author Lorenzo Moneta


#include "TRandom2.h"
#include "TF3.h"
#include "TError.h"
#include "Fit/BinData.h"
#include "Fit/Fitter.h"
#include "Math/WrappedMultiTF1.h"

void exampleFit3D() {

   const int n = 1000;
   double x[n], y[n], z[n], v[n];
   double ev = 0.1;

   // generate the data
   TRandom2 r;
   for (int i = 0; i < n; ++i) {
      x[i] = r.Uniform(0,10);
      y[i] = r.Uniform(0,10);
      z[i] = r.Uniform(0,10);
      v[i] = sin(x[i] ) + cos(y[i]) + z[i] + r.Gaus(0,ev);
   }

   // create a 3d binned data structure
   ROOT::Fit::BinData data(n,3);
   double xx[3];
   for(int i = 0; i < n; ++i) {
      xx[0] = x[i];
      xx[1] = y[i];
      xx[2] = z[i];
      // add the 3d-data coordinate, the predictor value (v[i])  and its errors
      data.Add(xx, v[i], ev);
   }

   TF3 * f3 = new TF3("f3","[0] * sin(x) + [1] * cos(y) + [2] * z",0,10,0,10,0,10);
   f3->SetParameters(2,2,2);
   ROOT::Fit::Fitter fitter;
   // wrapped the TF1 in a IParamMultiFunction interface for the Fitter class
   ROOT::Math::WrappedMultiTF1 wf(*f3,3);
   fitter.SetFunction(wf);
   //
   bool ret = fitter.Fit(data);
   if (ret) {
      const ROOT::Fit::FitResult & res = fitter.Result();
      // print result (should be around 1)
      res.Print(std::cout);
      // copy all fit result info (values, chi2, etc..) in TF3
      f3->SetFitResult(res);
      // test fit p-value (chi2 probability)
      double prob = res.Prob();
      if (prob < 1.E-2)
         Error("exampleFit3D","Bad data fit - fit p-value is %f",prob);
      else
         std::cout << "Good fit : p-value  = " << prob << std::endl;

   }
   else
      Error("exampleFit3D","3D fit failed");
}
