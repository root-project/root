// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TFcnAdapter_H_
#define ROOT_TFcnAdapter_H_

#ifndef ROOT_Minuit2_FCNGradientBase
#include "Minuit2/FCNGradientBase.h"
#endif

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

//___________________________________________________________
//
// Adapt the interface used in TMinuit (and the TVirtualFitter) for 
// passing the objective function in a Minuit2 interface 
// (ROOT::Minuit2::FCNGradientBase or ROOT::Minuit2::FCNBase)
//

class TFcnAdapter : public ROOT::Minuit2::FCNGradientBase {

public:

   TFcnAdapter(void (*fcn)(int&, double*, double&, double*, int)) : fFCN(fcn), fUp(1) {}

   virtual ~TFcnAdapter() {}

   const ROOT::Minuit2::FCNBase& Base() const {return *this;}
  
   double operator()(const std::vector<double>&) const;
   double Up() const {return fUp; }
  
   void SetErrorDef(double up) { fUp = up; }

   std::vector<double> Gradient(const std::vector<double>&) const;

   // forward interface
   double operator()(int npar, double* params,int iflag = 4) const;

private:

   void (*fFCN)(int&, double*, double&, double*, int);
   double fUp; 
   mutable std::vector<double> fGradCache; 

   ClassDef(TFcnAdapter,0)  // wrapper class implementing the Minuit2 interface for TMinuit2-like objective functions
};
#endif //ROOT_GFcnAdapter_H_
