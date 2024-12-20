/// \file
/// \ingroup tutorial_r
/// \notebook -nodraw
/// Simple example on how to use ROOT-R interface.
///
/// \macro_code
///
/// \author

#include "TMath.h"
#include "Math/PdfFunc.h"
#include "TMatrixD.h"
#include "TError.h"
#include <array>
#include <vector>
#include "TRInterface.h"

void example() {
   ROOT::R::TRInterface &r = ROOT::R::TRInterface::Instance();
   // print R version
   r.Execute("print(version$version.string)");

   // compute standard deviation of 1000 vector normal numbers

   double std_dev_r  = r.Eval("sd(rnorm(10000))");
   std::vector<double> v = r.Eval("rnorm(10000)");
   double std_dev_root = TMath::StdDev(v.begin(),v.end());
   std::cout << "standard deviation from R    = " << std_dev_r << std::endl;
   std::cout << "standard deviation from ROOT = " <<  std_dev_root << std::endl;
   if (!TMath::AreEqualAbs(std_dev_r,std_dev_root,0.1))
      Error("ROOT-R-Example","Different std-dev found");

   // use << to execute the R command instead of Execute
   r << "mat<-matrix(c(1,2,3,4,5,6),2,3,byrow=TRUE)";
   TMatrixD m = r["mat"];
   std::array<double,6> a = r.Eval("seq(1:6)");
   TMatrixD m2(2,3,a.data());

   if  (!(m==m2)) {
      Error("ROOT-R-Example","Different matrix  found");
      m.Print();
      m2.Print();
   }

   // example on how to pass ROOT objects to R
   std::vector<double> v_root{1,2,3,4,5,6,7,8};
   r["v"] = v_root;
   r << "v2<-seq(1:8)";
   bool isEqual = r.Eval("prod(v==v2)");
   if (!isEqual) {
      Error("ROOT-R-Example","Different vector created");
      r << "print(v)";
      r << "print(v2)";
   }

   // example on how to pass functions to R

   r["gaus_pdf"] = ROOT::Math::normal_pdf;

   r << "y<-gaus_pdf(0,1,1)";
   double value_r = r["y"];
   double value_root = ROOT::Math::normal_pdf(0,1,1);
   std::cout << "Function gaussian(0,1,1) evaluated in  R    = " << value_r << std::endl;
   std::cout << "Function gaussian(0,1,1) evaluated in ROOT  = " <<  value_root << std::endl;
   if (value_r != value_root)
      Error("ROOT-R-Example","Different function value found in r = %f and ROOT = %f", value_r, value_root);
}
