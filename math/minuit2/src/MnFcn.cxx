// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnFcn.h"
#include "Minuit2/MnUserTransformation.h"

namespace ROOT {

namespace Minuit2 {

double MnFcn::CallWithoutDoingTrafo(const MnAlgebraicVector &v) const
{
   // evaluate FCN converting from from MnAlgebraicVector to std::vector
   fNumCall++;
   return fFCN(std::vector<double>{v.Data(), v.Data() + v.size()});
}

double MnFcn::CallWithDoingTrafo(const MnAlgebraicVector &v) const
{
   // calling fTransform() like here was not thread safe because it was using a cached vector
   // return Fcn()( fTransform(v) );
   // make a new thread-safe implementation creating a vector each time
   // a bit slower few% in stressFit and 10% in Rosenbrock function but it is negligible in big fits

   // get first initial values of parameter (in case some one is fixed)
   std::vector<double> vpar(fTransform->InitialParValues().begin(), fTransform->InitialParValues().end());

   for (unsigned int i = 0; i < v.size(); i++) {
      vpar[fTransform->ExtOfInt(i)] = fTransform->Int2ext(i, v(i));
   }

   return CallWithTransformedParams(vpar);
}

// Calling the underlying function with the transformed parameters.
// For internal use in the Minuit2 implementation.
double MnFcn::CallWithTransformedParams(std::vector<double> const &vpar) const
{
   // call Fcn function transforming from a MnAlgebraicVector of internal values to a std::vector of external ones
   fNumCall++;

   return Fcn()(vpar);
}

} // namespace Minuit2

} // namespace ROOT
