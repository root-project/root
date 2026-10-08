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
 * Returns the bin width (or volume) of a list of observables, using their default binnings.
 * It can be used to normalise by bin width or to compute event densities. Using the extra
 * argument of the constructor, it can also return the inverse of the bin width (or volume).
 */

#include "RooBinWidthFunction.h"

#include "RooAbsBinning.h"
#include "RooAbsCategory.h"
#include "RooAbsLValue.h"
#include "RooAbsRealLValue.h"
#include "RooConstVar.h"
#include "RooCurve.h"
#include "RooGlobalFunc.h"
#include "RooHistFunc.h"
#include "RooFit/EvalContext.h"
#include "TBuffer.h"

#include <algorithm>
#include <stdexcept>

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

/// Construct from observables with default binnings. Real observables must be
/// RooAbsRealLValues; category observables contribute a unit bin width.
/// The observables are referenced, not owned, and must outlive this function.
/// \param name Name to identify the object.
/// \param title Title for e.g. plotting.
/// \param observables Observables whose bin widths are multiplied.
/// \param divideByBinWidth If true, return inverse bin volume.
RooBinWidthFunction::RooBinWidthFunction(const char *name, const char *title, const RooArgList &observables,
                                       bool divideByBinWidth)
   : RooAbsReal(name, title), _divideByBinWidth(divideByBinWidth)
{
   for (auto *arg : observables) {
      if (!dynamic_cast<RooAbsLValue *>(arg))
         throw std::invalid_argument("RooBinWidthFunction: observable " + std::string(arg->GetName()) +
                                     " must be an l-value with a binning");
      if (_observables.contains(*arg))
         throw std::invalid_argument("RooBinWidthFunction: duplicate observable " + std::string(arg->GetName()));
      _observables.add(*arg);
   }
}

/// Compatibility constructor. Only the live observables of histFunc are used;
/// its histogram contents and internal RooDataHist binning are irrelevant.
RooBinWidthFunction::RooBinWidthFunction(const char *name, const char *title, const RooHistFunc &histFunc,
                                       bool divideByBinWidth)
   : RooBinWidthFunction(name, title, RooArgList{histFunc.variables()}, divideByBinWidth)
{
}

/// Precompute volumes once, as RooDataHist did for the old implementation.
/// Value changes only select another bin; shape changes invalidate this table.
bool RooBinWidthFunction::updateCache() const
{
   bool rebuild = _binVolumes.empty() || isShapeDirty() || _binnings.size() != _observables.size();
   if (!rebuild) {
      for (std::size_t i = 0; i < _observables.size(); ++i) {
         auto &obs = dynamic_cast<const RooAbsLValue &>(_observables[i]);
         const auto *binning = obs.getBinningPtr(nullptr);
         if (_binnings[i] != binning || _binCounts[i] != obs.numBins() ||
             (binning && _binRanges[i] != std::make_pair(binning->lowBound(), binning->highBound()))) {
            rebuild = true;
            break;
         }
      }
   }
   if (!rebuild)
      return false;

   _binVolumes.assign(1, 1.);
   _binCounts.clear();
   _binnings.clear();
   _binRanges.clear();
   for (auto *arg : _observables) {
      auto &obs = dynamic_cast<const RooAbsLValue &>(*arg);
      int nBins = obs.numBins();
      if (nBins <= 0)
         throw std::invalid_argument("RooBinWidthFunction: observable " + std::string(arg->GetName()) + " has no bins");
      if (_binVolumes.size() > _binVolumes.max_size() / static_cast<std::size_t>(nBins))
         throw std::overflow_error("RooBinWidthFunction: too many bins");
      std::vector<double> volumes;
      volumes.reserve(_binVolumes.size() * nBins);
      for (double volume : _binVolumes) {
         for (int bin = 0; bin < nBins; ++bin)
            volumes.push_back(volume * obs.getBinWidth(bin));
      }
      _binVolumes.swap(volumes);
      _binCounts.push_back(nBins);
      const auto *binning = obs.getBinningPtr(nullptr);
      _binnings.push_back(binning);
      _binRanges.emplace_back(binning ? binning->lowBound() : 0., binning ? binning->highBound() : 0.);
   }
   clearShapeDirty();
   return true;
}

// RooRealVar::setBinning() replaces the binning without sending a shape-dirty
// notification. Check the geometry before reusing RooAbsReal's cached value.
// This also handles the changing range of a RooLinearVar's binning adaptor.
double RooBinWidthFunction::getValV(const RooArgSet *nset) const
{
   if (_enabled && updateCache())
      const_cast<RooBinWidthFunction *>(this)->setValueDirty();
   return RooAbsReal::getValV(nset);
}

/// Compute current bin of observable, and return its volume or inverse volume, depending
/// on configuration chosen in the constructor.
/// If the bin is not valid, return a volume of 1.
double RooBinWidthFunction::evaluate() const {
  if(!_enabled) return 1.;
  updateCache();
  std::size_t index = 0;
  for (std::size_t i = 0; i < _observables.size(); ++i) {
    auto &arg = _observables[i];
    if (!arg.inRange(nullptr))
      return 1.;
    int bin = dynamic_cast<const RooAbsLValue &>(arg).getBin();
    if (bin < 0 || bin >= _binCounts[i])
      return 1.;
    index = index * _binCounts[i] + bin;
  }
  double volume = _binVolumes[index];
  return _divideByBinWidth ? 1./volume : volume;
}


/// Compute bin index for all values of the observable(s) in `evalData`, and return their volumes or inverse volumes, depending
/// on the configuration chosen in the constructor.
/// If a bin is not valid, return a volume of 1.
void RooBinWidthFunction::doEval(RooFit::EvalContext &ctx) const
{
   auto output = ctx.output();
   if (!_enabled) {
      std::fill(output.begin(), output.end(), 1.);
      return;
   }
   updateCache();
   std::vector<std::span<const double>> inputs;
   for (auto *arg : _observables)
      inputs.push_back(ctx.at(arg));

   for (std::size_t event = 0; event < output.size(); ++event) {
      std::size_t index = 0;
      bool valid = true;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
         double value = inputs[i][inputs[i].size() == 1 ? 0 : event];
         int bin;
         if (auto *binning = _binnings[i]) {
            if (value < binning->lowBound() || value > binning->highBound()) {
               valid = false;
               break;
            }
            bin = binning->binNumber(value);
         } else {
            // Categories have unit volume in every state. Their numeric state
            // indices need not be consecutive bin numbers.
            auto &category = static_cast<const RooAbsCategory &>(_observables[i]);
            if (!category.hasIndex(static_cast<int>(value))) {
               valid = false;
               break;
            }
            bin = 0;
         }
         if (bin < 0 || bin >= _binCounts[i]) {
            valid = false;
            break;
         }
         index = index * _binCounts[i] + bin;
      }
      double volume = valid ? _binVolumes[index] : 1.;
      output[event] = _divideByBinWidth ? 1. / volume : volume;
   }
}

std::list<double> *RooBinWidthFunction::binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   auto *arg = dynamic_cast<RooAbsRealLValue *>(_observables.find(obs.GetName()));
   if (!arg)
      return nullptr;
   const auto &binning = arg->getBinning();
   auto hint = std::make_unique<std::list<double>>();
   const double delta = (xhi - xlo) * 1e-8;
   const double *boundaries = binning.array();
   for (int i = 0; i < binning.numBoundaries(); ++i) {
      if (boundaries[i] > xlo - delta && boundaries[i] < xhi + delta)
         hint->push_back(boundaries[i]);
   }
   return hint.release();
}

std::list<double> *RooBinWidthFunction::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   auto *arg = dynamic_cast<RooAbsRealLValue *>(_observables.find(obs.GetName()));
   if (!arg)
      return nullptr;
   const auto &binning = arg->getBinning();
   return RooCurve::plotSamplingHintForBinBoundaries(
      {binning.array(), static_cast<std::size_t>(binning.numBoundaries())}, xlo, xhi);
}

/// Rebuild proxy registration after schema evolution replaces the v1 proxy.
void RooBinWidthFunction::Streamer(TBuffer &buffer)
{
   if (buffer.IsReading()) {
      buffer.ReadClassBuffer(RooBinWidthFunction::Class(), this);
      finalizeIO();
   } else {
      buffer.WriteClassBuffer(RooBinWidthFunction::Class(), this);
   }
}

// The old proxy can be read before its RooHistFunc has finished streaming
// (for example, when two bin-width functions share a histogram). Complete the
// conversion in the workspace's second pass, when all observable lists exist.
void RooBinWidthFunction::ioStreamerPass2()
{
   RooAbsReal::ioStreamerPass2();
   finalizeIO();
}

void RooBinWidthFunction::finalizeIO()
{
   RooHistFunc *hist = nullptr;
   for (auto *server : servers()) {
      if ((hist = dynamic_cast<RooHistFunc *>(server)))
         break;
   }
   if (hist && !hist->variables().empty()) {
      if (_observables.empty())
         _observables.RooArgList::add(hist->variables());
      removeServer(*hist, true);
      // The old constructor already registered the observable value servers.
      // Enable shape propagation without adding duplicate server references.
      for (auto *arg : _observables)
         changeServer(*arg, true, true);
   }
   // RooAbsArg's second I/O pass restores the old proxy list. Replace it with
   // the new proxy, including when reading the current class layout.
   _proxyList.Clear();
   registerProxy(_observables);
   _binVolumes.clear();
   setValueDirty();
   setShapeDirty();
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
