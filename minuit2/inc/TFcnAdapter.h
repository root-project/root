// @(#)root/minuit2:$Name:  $:$Id: TFcnAdapter.h,v 1.3 2006/01/25 12:20:49 moneta Exp $
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

   TFcnAdapter(void (*fcn)(int&, double*, double&, double*, int)) : fFCN(fcn) {}

   ~TFcnAdapter() {}

   const ROOT::Minuit2::FCNBase& Base() const {return *this;}
  
   virtual double operator()(const std::vector<double>&) const;
   virtual double Up() const {return 1;}
  
   virtual std::vector<double> Gradient(const std::vector<double>&) const;

   // forward interface
   virtual double operator()(int npar, double* params,int iflag = 4) const;

private:

   void (*fFCN)(int&, double*, double&, double*, int);
};
#endif //ROOT_GFcnAdapter_H_
