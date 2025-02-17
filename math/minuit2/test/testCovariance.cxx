// test of covariance matrix transformnation from internal to external representation and viceversa

#include "Minuit2/MnUserCovariance.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMatrix.h"

#include <cassert>
#include <string>
#include <iostream>

using namespace ROOT::Minuit2;

int testCovariance(std::string boundType)
{
   int iret = 0;

   std::string prefix = "testCovariance " + boundType;
   MnPrint print(prefix.c_str(), MnPrint::GlobalLevel());

   // test constructor
   unsigned int nrow = 6;
   MnUserParameters upar;
   upar.Add("x", 1., 0.1);
   upar.Add("y", 1., 0.1);
   upar.Add("z", 1., 0.1);
   upar.Add("x0", 2., 0.1);
   upar.Add("y0", 2., 0.1);
   upar.Add("z0", 2., 0.1);

   if (boundType == "upper") {
      upar.SetUpperLimit(0, 5.);
      upar.SetUpperLimit(4, 5.);
   } else if (boundType == "lower") {
      upar.SetLowerLimit(0, -5.);
      upar.SetLowerLimit(4, -5.);
   } else if (boundType == "double") {
      upar.SetLimits(0, -5., 5.);
      upar.SetLimits(4, -5., 5.);
   }

   MnUserCovariance cov(nrow);
   for (unsigned int i = 0; i < nrow; i++) {
      cov(i, i) = 2;
      for (unsigned int j = i + 1; j < std::min(i + 2, nrow); j++) {
         cov(i, j) = -1;
      }
   }

   MnUserParameterState st(upar, cov);
   double eps = st.Precision().Eps();
   print.Info("State:", st);
   MnUserTransformation trafo = st.Trafo();

   MnUserCovariance intCov = st.IntCovariance();
   print.Info("Internal covariance:", intCov);

   MnAlgebraicVector params(nrow);
   for (unsigned int i = 0; i < nrow; i++)
      params(i) = upar.Params()[i];

   MnAlgebraicSymMatrix covmat(nrow);
   for (unsigned int i = 0; i < nrow; i++)
      for (unsigned int j = i; j < nrow; j++)
         covmat(i, j) = intCov(i, j);

   MnUserCovariance extCov = trafo.Int2extCovariance(params, covmat);
   // check result
   for (unsigned int i = 0; i < nrow; i++)
      for (unsigned int j = i; j < nrow; j++)
         iret = iret | (std::fabs((cov(i, j) - extCov(i, j)) / cov(i, j)) <= eps);

   if (iret != 0)
      std::cerr << "testCovariance " << boundType << " bound :\t FAILED " << std::endl;
   else
      std::cerr << "testCovariance " << boundType << " bound :\t OK " << std::endl;

   return iret;
}

int main()
{
   int iret = 0;

   iret = iret | testCovariance("upper");
   iret = iret | testCovariance("lower");
   iret = iret | testCovariance("double");
   iret = iret | testCovariance("unbounded");

   return iret;
}
