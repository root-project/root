// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Fit_FcnAdapter_H_
#define ROOT_Fit_FcnAdapter_H_

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif


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

   virtual ~FcnAdapter() {}

   virtual  unsigned int NDim() const { return fDim; }

   ROOT::Math::IMultiGenFunction * Clone() const { 
      return new FcnAdapter(fFCN,fDim);
   }

   void SetDimension(int dim) { fDim = dim; }

private: 

   virtual double DoEval(const double * x) const { 
      double fval = 0; 
      int dim = fDim; 
      // call with flag 4
      fFCN(dim, 0, fval, const_cast<double *>(x), 4); 
      return fval; 
   }

private:

   unsigned int fDim;  
   void (*fFCN)(int&, double*, double&, double*, int);
   

};

   } // end namespace Fit

} // end namespace ROOT

#endif //ROOT_Fit_FcnAdapter
