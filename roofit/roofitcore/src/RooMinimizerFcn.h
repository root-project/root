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
#include "Fit/ParameterSettings.h"
#include "Fit/FitResult.h"

#include "RooAbsReal.h"
#include "RooArgList.h"

#include <fstream>
#include <vector>

#include "RooAbsMinimizerFcn.h"

template <typename T>
class TMatrixTSym;
using TMatrixDSym = TMatrixTSym<double>;

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
   RooAbsReal *_funct;
   std::unique_ptr<ROOT::Math::IBaseFunctionMultiDim> _multiGenFcn;
};

#endif
