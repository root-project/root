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

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooFitDriver.h"

#include "Math/IFunction.h"
#include "Fit/ParameterSettings.h"
#include "Fit/FitResult.h"

#include "RooArgList.h"

#include <fstream>
#include <vector>

#include <RooAbsMinimizerFcn.h>

template<typename T> class TMatrixTSym;
using TMatrixDSym = TMatrixTSym<double>;

// forward declaration
class RooMinimizer;

class RooMinimizerFcn : public RooAbsMinimizerFcn, public ROOT::Math::IBaseFunctionMultiDim {

public:
   RooMinimizerFcn(Function && funct, RooMinimizer *context, bool verbose = false);
   RooMinimizerFcn(const RooMinimizerFcn &other);
   virtual ~RooMinimizerFcn();

   ROOT::Math::IBaseFunctionMultiDim *Clone() const override;
   unsigned int NDim() const override { return getNDim(); }

   inline std::string const& getFunctionName() const override { return _funct.name; }
   inline std::string const& getFunctionTitle() const override { return _funct.title; }

   Function & funct() override { return _funct; }

private:
   double DoEval(const double *x) const override;

   Function _funct;
};

#endif
