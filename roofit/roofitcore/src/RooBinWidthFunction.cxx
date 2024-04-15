// Author Stephan Hageboeck, CERN, 10/2020
/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2020, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


/**
 * \class RooBinWidthFunction
 *  \ingroup Roofitcore
 *
 * Returns the bin width (or volume) given a RooHistFunc.
 * It can be used to normalise by bin width or to compute event densities. Using the extra
 * argument of the constructor, it can also return the inverse of the bin width (or volume).
 */

#include "RooBinWidthFunction.h"

#include "RooConstVar.h"
#include "RooDataHist.h"
#include "RooGlobalFunc.h"

bool RooBinWidthFunction::_enabled = true;

/// Globally enable bin-width corrections by this class.
void RooBinWidthFunction::enableClass() {
  _enabled = true;
}

/// Returns `true` if bin-width corrections by this class are globally enabled, `false` otherwise.
bool RooBinWidthFunction::isClassEnabled() {
  return _enabled;
}

/// Globally disable bin-width corrections by this class.
void RooBinWidthFunction::disableClass() {
  _enabled = false;
}

/// Create an instance.
/// \param name Name to identify the object.
/// \param title Title for e.g. plotting.
/// \param histFunc RooHistFunc object whose bin widths should be returned.
/// \param divideByBinWidth If true, return inverse bin width.
RooBinWidthFunction::RooBinWidthFunction(const char *name, const char *title, const RooHistFunc &histFunc, bool divideByBinWidth)
   : RooAbsReal(name, title),
     _histFunc("HistFuncForBinWidth", "Handle to a RooHistFunc, whose bin volumes should be returned.", this, histFunc,
               /*valueServer=*/false, /*shapeServer=*/false),
     _divideByBinWidth(divideByBinWidth)
{
   // The RooHistFunc is only used to access this histogram observables in a
   // convenient way. That's why this proxy is not "serving" this
   // RooBinWidthFunction in any way (see proxy constructor arguments in the
   // initializer list above).
   //
   // However, the variables of the histFunc **need to be** value servers,
   // because the width of the current bin depends on the values of the
   // observables:
   for (RooAbsArg * server : histFunc.servers()) {
      addServer(*server, /*valueServer=*/true, /*shapeServer=*/false);
   }
   // The reason why we can't simply use the histFunc as an "indirect proxy" is
   // the way HistFactory is implemented. The same RooBinWidthFunction is used
   // for all samples (e.g. signal and backgrounds), but uses the RooHistFunc
   // of only one of the samples (this is okay because the binnings for all
   // samples in the template histogram stack is the same). This entangling of
   // the computation graph for the different samples messes up the component
   // selection when plotting only some samples with
   // `plotOn(..., RooFit::Components(...))`.
}

/// Compute current bin of observable, and return its volume or inverse volume, depending
/// on configuration chosen in the constructor.
/// If the bin is not valid, return a volume of 1.
double RooBinWidthFunction::evaluate() const {
  if(!_enabled) return 1.;
  const RooDataHist& dataHist = _histFunc->dataHist();
  const auto idx = _histFunc->getBin();
  auto volumes = dataHist.binVolumes(0, dataHist.numEntries());
  const double volume = idx >= 0 ? volumes[idx] : 1.;

  return _divideByBinWidth ? 1./volume : volume;
}


/// Compute bin index for all values of the observable(s) in `evalData`, and return their volumes or inverse volumes, depending
/// on the configuration chosen in the constructor.
/// If a bin is not valid, return a volume of 1.
void RooBinWidthFunction::doEval(RooFit::EvalContext &ctx) const
{
   std::span<double> output = ctx.output();
   const RooDataHist &dataHist = _histFunc->dataHist();
   std::vector<Int_t> bins = _histFunc->getBins(ctx);
   auto volumes = dataHist.binVolumes(0, dataHist.numEntries());

   if (!_enabled) {
      for (std::size_t i = 0; i < bins.size(); ++i) {
         output[i] = 1.;
      }
   } else {
      if (_divideByBinWidth) {
         for (std::size_t i = 0; i < bins.size(); ++i) {
            output[i] = bins[i] >= 0 ? 1. / volumes[bins[i]] : 1.;
         }
      } else {
         for (std::size_t i = 0; i < bins.size(); ++i) {
            output[i] = bins[i] >= 0 ? volumes[bins[i]] : 1.;
         }
      }
   }
}


std::unique_ptr<RooAbsArg>
RooBinWidthFunction::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   // If this is a binned likelihood, the pdf values can be directly
   // interpreted as yields for Poisson terms in the NLL, and it doesn't make
   // sense to divide them by the bin width to get a probability density. The
   // NLL would only have to multiply by the bin with again.
   if (ctx.binnedLikelihoodMode()) {
      auto newArg = std::unique_ptr<RooAbsReal>{static_cast<RooAbsReal *>(RooFit::RooConst(1.0).Clone())};
      ctx.markAsCompiled(*newArg);
      // To propagate the information to the NLL that the pdf values can
      // directly be interpreted as yields.
      ctx.setBinWidthFuncFlag(true);
      return newArg;
   }
   return RooAbsReal::compileForNormSet(normSet, ctx);
}
