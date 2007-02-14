// @(#)root/minuit2:$Name:  $:$Id: TFcnAdapter.h,v 1.4 2006/07/05 08:32:39 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TFcnAdapter_H_
#define ROOT_TFcnAdapter_H_

#include "Minuit2/FCNGradientBase.h"

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
};
#endif //ROOT_GFcnAdapter_H_
