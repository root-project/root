// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef MN_GaussFunction_H_
#define MN_GaussFunction_H_

#define _USE_MATH_DEFINES
#include <math.h>

namespace ROOT {

   namespace Minuit2 {


class GaussFunction {

public:
  
  GaussFunction(double mean, double sig, double constant) : 
    fMean(mean), fSigma(sig), fConstant(constant) {}

  ~GaussFunction() {}

  double m() const {return fMean;}
  double s() const {return fSigma;}
  double c() const {return fConstant;}

  double operator()(double x) const {
    
    return c()*exp(-0.5*(x-m())*(x-m())/(s()*s()))/(sqrt(2.*M_PI)*s());
  }

private:
  
  double fMean;
  double fSigma;
  double fConstant;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif // MN_GaussFunction_H_
