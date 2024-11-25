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

namespace RooFit {
namespace TestStatistics {

class LikelihoodSerial : public LikelihoodWrapper {
public:
   LikelihoodSerial(std::shared_ptr<RooAbsL> _likelihood,
                    std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean, SharedOffset offset);

   void initVars();

   void evaluate() override;
   inline ROOT::Math::KahanSum<double> getResult() const override { return result; }

private:
   ROOT::Math::KahanSum<double> result;

   RooArgList _vars;     ///< Variables
   RooArgList _saveVars; ///< Copy of variables
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_LikelihoodSerial
