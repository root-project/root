/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl   *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_MINIMIZER_FCN
#define ROO_MINIMIZER_FCN

#include "Math/IFunction.h"

#include "RooAbsReal.h"
#include "RooArgList.h"

#include <fstream>
#include <vector>

#include "RooAbsMinimizerFcn.h"

// forward declaration
class RooMinimizer;

class RooMinimizerFcn : public RooAbsMinimizerFcn {

public:
   RooMinimizerFcn(RooAbsReal *funct, RooMinimizer *context);

   std::string getFunctionName() const override;
   std::string getFunctionTitle() const override;

   void setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) override;

   void setOffsetting(bool flag) override;
   ROOT::Math::IMultiGenFunction *getMultiGenFcn() override { return _multiGenFcn.get(); }

   double operator()(const double *x) const;
   void evaluateGradient(const double *x, double *out) const;

private:
   RooAbsReal *_funct = nullptr;
   std::unique_ptr<ROOT::Math::IBaseFunctionMultiDim> _multiGenFcn;
   mutable std::vector<double> _gradientOutput;
};

#endif
