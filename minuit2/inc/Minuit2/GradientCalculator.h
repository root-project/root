// @(#)root/minuit2:$Name:  $:$Id: GradientCalculator.h,v 1.2.6.2 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_GradientCalculator
#define ROOT_Minuit2_GradientCalculator

namespace ROOT {

   namespace Minuit2 {


class MinimumParameters;
class FunctionGradient;

class GradientCalculator {

public:
  
  virtual ~GradientCalculator() {}

  virtual FunctionGradient operator()(const MinimumParameters&) const = 0;

  virtual FunctionGradient operator()(const MinimumParameters&,
				      const FunctionGradient&) const = 0;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_GradientCalculator
