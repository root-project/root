// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOT_ROOFIT_LikelihoodSerial
#define ROOT_ROOFIT_LikelihoodSerial

#include <TestStatistics/LikelihoodWrapper.h>
#include "RooArgList.h"

#include "Math/MinimizerOptions.h"

#include <map>

namespace RooFit {
namespace TestStatistics {

class LikelihoodSerial : public LikelihoodWrapper {
public:
   LikelihoodSerial(std::shared_ptr<RooAbsL> likelihood, std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean/*, RooMinimizer *minimizer*/);
   inline LikelihoodSerial *clone() const override { return new LikelihoodSerial(*this); }

   void initVars();

   void evaluate() override;
   inline double getResult() const override { return result; }

private:
   double result = 0;
   double carry = 0;

   RooArgList _vars;      // Variables
   RooArgList _saveVars;  // Copy of variables

   LikelihoodType likelihood_type;
};

}
}

#endif // ROOT_ROOFIT_LikelihoodSerial
