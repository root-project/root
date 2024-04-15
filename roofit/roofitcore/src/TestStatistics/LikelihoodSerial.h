/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_LikelihoodSerial
#define ROOT_ROOFIT_LikelihoodSerial

#include <RooFit/TestStatistics/LikelihoodWrapper.h>
#include "RooArgList.h"

#include "Math/MinimizerOptions.h"

#include <map>

namespace RooFit {
namespace TestStatistics {

class LikelihoodSerial : public LikelihoodWrapper {
public:
   LikelihoodSerial(std::shared_ptr<RooAbsL> likelihood,
                    std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean);
   inline LikelihoodSerial *clone() const override { return new LikelihoodSerial(*this); }

   void initVars();

   void evaluate() override;
   inline ROOT::Math::KahanSum<double> getResult() const override { return result; }

private:
   ROOT::Math::KahanSum<double> result;

   RooArgList _vars;     ///< Variables
   RooArgList _saveVars; ///< Copy of variables

   LikelihoodType likelihood_type;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_LikelihoodSerial
