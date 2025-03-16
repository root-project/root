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

// Calling the underlying function with the transformed parameters.
// For internal use in the Minuit2 implementation.
double MnFcn::CallWithTransformedParams(std::vector<double> const &vpar) const
{
   // call Fcn function transforming from a MnAlgebraicVector of internal values to a std::vector of external ones
   fNumCall++;

   return Fcn()(vpar);
}

MnFcnCaller::MnFcnCaller(const MnFcn &mfcn) : fMfcn{mfcn}, fDoInt2ext{static_cast<bool>(mfcn.Trafo())}
{
   if (!fDoInt2ext)
      return;

   MnUserTransformation const &transform = *fMfcn.Trafo();

   // get first initial values of parameter (in case some one is fixed)
   fVpar.assign(transform.InitialParValues().begin(), transform.InitialParValues().end());
}

double MnFcnCaller::operator()(const MnAlgebraicVector &v)
{
   if (!fDoInt2ext)
      return fMfcn.CallWithoutDoingTrafo(v);

   MnUserTransformation const &transform = *fMfcn.Trafo();

   bool firstCall = fLastInput.size() != v.size();

   fLastInput.resize(v.size());

   for (unsigned int i = 0; i < v.size(); i++) {
      if (firstCall || fLastInput[i] != v(i)) {
         fVpar[transform.ExtOfInt(i)] = transform.Int2ext(i, v(i));
         fLastInput[i] = v(i);
      }
   }

   return fMfcn.CallWithTransformedParams(fVpar);
}

} // namespace Minuit2

} // namespace ROOT
