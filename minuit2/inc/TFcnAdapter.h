// @(#)root/minuit2:$Name:  $:$Id: TFcnAdapter.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TFcnAdapter_H_
#define ROOT_TFcnAdapter_H_

#include "Minuit/FCNGradientBase.h"

class TFcnAdapter : public FCNGradientBase {

public:

  TFcnAdapter(void (*fcn)(int&, double*, double&, double*, int)) : fFCN(fcn) {}

  ~TFcnAdapter() {}

  const FCNBase& base() const {return *this;}
  
  virtual double operator()(const std::vector<double>&) const;
  virtual double up() const {return 1;}
  
  virtual std::vector<double> gradient(const std::vector<double>&) const;

  // forward interface
  virtual double operator()(int npar, double* params) const;

private:

  void (*fFCN)(int&, double*, double&, double*, int);
};
#endif //ROOT_GFcnAdapter_H_
