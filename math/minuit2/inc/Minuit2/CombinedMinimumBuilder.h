// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_CombinedMinimumBuilder
#define ROOT_Minuit2_CombinedMinimumBuilder

#include "Minuit2/MinimumBuilder.h"
#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/SimplexMinimizer.h"

namespace ROOT {

namespace Minuit2 {

class CombinedMinimumBuilder : public MinimumBuilder {

public:
   CombinedMinimumBuilder() : fVMMinimizer(VariableMetricMinimizer()), fSimplexMinimizer(SimplexMinimizer()) {}

   ~CombinedMinimumBuilder() override {}

   FunctionMinimum Minimum(const MnFcn &, const GradientCalculator &, const MinimumSeed &, const MnStrategy &,
                                   unsigned int, double) const override;

   // re-implement setter of base class. Need also to store in the base class for consistency
   void SetPrintLevel(int level) override
   {
      MinimumBuilder::SetPrintLevel(level);
      fVMMinimizer.Builder().SetPrintLevel(level);
      fSimplexMinimizer.Builder().SetPrintLevel(level);
   }
   void SetStorageLevel(int level) override
   {
      MinimumBuilder::SetStorageLevel(level);
      fVMMinimizer.Builder().SetStorageLevel(level);
      fSimplexMinimizer.Builder().SetStorageLevel(level);
   }

   // set trace object (user manages it)
   void SetTraceObject(MnTraceObject &obj) override
   {
      MinimumBuilder::SetTraceObject(obj);
      fVMMinimizer.Builder().SetTraceObject(obj);
      fSimplexMinimizer.Builder().SetTraceObject(obj);
   }

private:
   VariableMetricMinimizer fVMMinimizer;
   SimplexMinimizer fSimplexMinimizer;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_CombinedMinimumBuilder
