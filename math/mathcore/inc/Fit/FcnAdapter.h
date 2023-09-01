// @(#)root/mathcore:$Id$
// Author: L. Moneta    10/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Fit_FcnAdapter_H_
#define ROOT_Fit_FcnAdapter_H_

#include "Math/IFunction.h"


//___________________________________________________________
//
// Adapt the interface used in TMinuit (and the TVirtualFitter) for
// passing the objective function in a IFunction  interface
// (ROOT::Math::IMultiGenFunction)
//

namespace ROOT {

   namespace Fit {

class FcnAdapter : public ROOT::Math::IMultiGenFunction {

public:

   FcnAdapter(void (*fcn)(int&, double*, double&, double*, int ), int dim = 0) :
      fDim(dim),
      fFCN(fcn)
   {}

   ~FcnAdapter() override {}

    unsigned int NDim() const override { return fDim; }

   ROOT::Math::IMultiGenFunction * Clone() const override {
      return new FcnAdapter(fFCN,fDim);
   }

   void SetDimension(int dim) { fDim = dim; }

private:

   double DoEval(const double * x) const override {
      double fval = 0;
      int dim = fDim;
      // call with flag 4
      fFCN(dim, nullptr, fval, const_cast<double *>(x), 4);
      return fval;
   }

private:

   unsigned int fDim;
   void (*fFCN)(int&, double*, double&, double*, int);

};

   } // end namespace Fit

} // end namespace ROOT

#endif //ROOT_Fit_FcnAdapter
