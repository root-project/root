/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooHistFactorySample
\ingroup HistFactory

One sample of a HistFactory channel, i.e. of a RooHistFactoryPdf: a histogram whose bin contents are modified by
parameters. It corresponds to a sample of the `histfactory_dist` in the HS3 standard. The value in bin \f$ b \f$ is

\f[
  \nu_b = \left( c_b + \sum_i I_i(\alpha_i; l_{i,b}, c_b, h_{i,b}) \right)
          \prod_j \gamma_{j,b} \; \prod_k \eta_k(\theta_k) \prod_m \mu_m ,
\f]

with the nominal contents \f$ c_b \f$, the HistoSys variations \f$ l_{i,b} \f$ and \f$ h_{i,b} \f$, the parameters
\f$ \gamma_{j,b} \f$ of the bin-wise modifiers (staterror, shapesys, shapefactor), the NormSys factors
\f$ \eta_k \f$ and the NormFactors \f$ \mu_m \f$. The interpolations are the ones of PiecewiseInterpolation (HistoSys)
and FlexibleInterpVar (NormSys), and if there is a HistoSys, the interpolated bin contents are clamped at zero.

The value of the function is the density in the bin of the current observable values, i.e. \f$ \nu_b \f$ divided by
the bin volume. In the binned likelihood, the sample is evaluated as the yield \f$ \nu_b \f$ itself.

In contrast to the HistFactory models before ROOT 6.44, which consist of several RooFit objects for each histogram,
a sample is a single object. The histogram data is shared between the copies of a sample.
**/

#include <RooStats/HistFactory/RooHistFactorySample.h>

#include "HistFactoryInterpolationCodeUtils.h"

#include <RooCurve.h>
#include <RooFit/Detail/MathFuncs.h>
#include <RooFit/Detail/NormalizationHelpers.h>
#include <RooFit/EvalContext.h>
#include <RooMsgService.h>
#include <RooRealVar.h>

#include <TBuffer.h>
#include <TClass.h>

#include <algorithm>
#include <stdexcept>

namespace {

int checkedIndex(RooArgList &params, RooAbsReal *param, std::string const &sample, std::string const &modifier)
{
   if (!param) {
      throw std::invalid_argument("RooHistFactorySample: the modifier \"" + modifier + "\" of sample \"" + sample +
                                  "\" has no parameter");
   }
   int index = params.index(param);
   if (index < 0) {
      params.add(*param);
      index = params.size() - 1;
   }
   return index;
}

void checkSize(std::size_t size, std::size_t nBins, std::string const &sample, std::string const &what)
{
   if (size != nBins) {
      throw std::invalid_argument("RooHistFactorySample: " + what + " of sample \"" + sample + "\" has " +
                                  std::to_string(size) + " bins instead of " + std::to_string(nBins));
   }
}

double binWidth(RooHistFactorySample::Data const &data, std::size_t iDim, std::size_t iBin)
{
   std::vector<double> const &edges = data.edges[iDim];
   if (data.uniform[iDim]) {
      // The same as RooUniformBinning::binWidth()
      return (edges.back() - edges.front()) / (edges.size() - 1);
   }
   return edges[iBin + 1] - edges[iBin];
}

std::size_t binInDim(RooHistFactorySample::Data const &data, std::size_t iDim, double x)
{
   std::vector<double> const &edges = data.edges[iDim];
   const int nBins = edges.size() - 1;
   if (data.uniform[iDim]) {
      // The same as RooUniformBinning::binNumbers()
      const double oneOverW = 1. / ((edges.back() - edges.front()) / nBins);
      return x >= edges.back() ? nBins - 1 : std::min(nBins - 1, std::max(0, int((x - edges.front()) * oneOverW)));
   }
   return RooFit::Detail::MathFuncs::binNumber(x, 1.0, edges.data(), edges.size(), nBins - 1, 0);
}

} // namespace

/// Create a sample.
/// \param name The name of the RooFit object.
/// \param title The title of the RooFit object.
/// \param observables The observables, which need to be RooRealVars. Their binning defines the bins of the sample.
/// \param spec The bin contents and the modifiers of the sample.
RooHistFactorySample::RooHistFactorySample(const char *name, const char *title, RooArgList const &observables,
                                           Spec const &spec)
   : RooAbsReal(name, title), _obs("!obs", "observables", this), _params("!params", "parameters", this)
{
   auto data = std::make_shared<Data>();
   data->sampleName = spec.sampleName;

   std::size_t nBins = 1;
   for (RooAbsArg *arg : observables) {
      auto *obs = dynamic_cast<RooRealVar *>(arg);
      if (!obs) {
         throw std::invalid_argument(std::string("RooHistFactorySample: the observable \"") + arg->GetName() +
                                     "\" of sample \"" + spec.sampleName + "\" is not a RooRealVar");
      }
      _obs.add(*obs);
      RooAbsBinning const &binning = obs->getBinning();
      // Each call to RooBinning::array() reallocates the array.
      double const *boundaries = binning.array();
      data->edges.emplace_back(boundaries, boundaries + binning.numBoundaries());
      data->uniform.push_back(binning.isUniform());
      // The array of a uniform binning is computed from the low bound and the bin width, so the high bound has to be
      // set explicitly to find the bins exactly like RooUniformBinning.
      if (binning.isUniform()) {
         data->edges.back().back() = binning.highBound();
      }
      nBins *= binning.numBins();
   }

   checkSize(spec.contents.size(), nBins, spec.sampleName, "the nominal histogram");
   data->contents = spec.contents;
   if (!spec.errors.empty()) {
      checkSize(spec.errors.size(), nBins, spec.sampleName, "the histogram of the statistical uncertainties");
      data->errors = spec.errors;
   }

   for (NormFactor const &mod : spec.normFactors) {
      data->normFactorNames.push_back(mod.name);
      data->normFactorParams.push_back(checkedIndex(_params, mod.param, spec.sampleName, mod.name));
   }

   for (NormSys const &mod : spec.normSys) {
      data->normSysNames.push_back(mod.name);
      data->normSysParams.push_back(checkedIndex(_params, mod.param, spec.sampleName, mod.name));
      data->normSysLo.push_back(mod.lo);
      data->normSysHi.push_back(mod.hi);
      data->normSysCodes.push_back(mod.interpCode);
   }

   for (HistoSys const &mod : spec.histoSys) {
      checkSize(mod.lo.size(), nBins, spec.sampleName, "the low variation of \"" + mod.name + "\"");
      checkSize(mod.hi.size(), nBins, spec.sampleName, "the high variation of \"" + mod.name + "\"");
      data->histoSysNames.push_back(mod.name);
      data->histoSysParams.push_back(checkedIndex(_params, mod.param, spec.sampleName, mod.name));
      data->histoSysCodes.push_back(mod.interpCode);
      data->histoSysLo.insert(data->histoSysLo.end(), mod.lo.begin(), mod.lo.end());
      data->histoSysHi.insert(data->histoSysHi.end(), mod.hi.begin(), mod.hi.end());
   }

   for (BinFactor const &mod : spec.binFactors) {
      checkSize(mod.params.size(), nBins, spec.sampleName, "the parameters of \"" + mod.name + "\"");
      data->binFactorNames.push_back(mod.name);
      data->binFactorTypes.push_back(static_cast<int>(mod.type));
      for (RooAbsArg *param : mod.params) {
         data->binFactorParams.push_back(
            checkedIndex(_params, dynamic_cast<RooAbsReal *>(param), spec.sampleName, mod.name));
      }
   }

   _data = std::move(data);

   // Validate the interpolation codes like the classes that implement them.
   for (std::size_t i = 0; i < _data->normSysCodes.size(); ++i) {
      if (_data->normSysCodes[i] < 0 || _data->normSysCodes[i] > 5) {
         throw std::invalid_argument("RooHistFactorySample: unknown NormSys interpolation code " +
                                     std::to_string(_data->normSysCodes[i]) + " for \"" + _data->normSysNames[i] +
                                     "\"");
      }
   }
   for (std::size_t i = 0; i < _data->histoSysCodes.size(); ++i) {
      if (_data->histoSysCodes[i] < 0 || _data->histoSysCodes[i] > 6 || _data->histoSysCodes[i] == 3) {
         throw std::invalid_argument("RooHistFactorySample: unknown HistoSys interpolation code " +
                                     std::to_string(_data->histoSysCodes[i]) + " for \"" + _data->histoSysNames[i] +
                                     "\"");
      }
   }
}

RooHistFactorySample::RooHistFactorySample(const RooHistFactorySample &other, const char *name)
   : RooAbsReal(other, name),
     _obs("!obs", this, other._obs),
     _params("!params", this, other._params),
     _data(other._data),
     _yieldsMode(other._yieldsMode)
{
}

/// The volume of a bin, i.e. the product of its widths in all observables.
double RooHistFactorySample::binVolume(std::size_t iBin) const
{
   Data const &data = *_data;
   // Like RooDataHist, the bin volume is accumulated starting from the first observable.
   double volume = 1.0;
   std::size_t stride = data.numBins();
   for (std::size_t iDim = 0; iDim < data.edges.size(); ++iDim) {
      stride /= data.edges[iDim].size() - 1;
      volume *= binWidth(data, iDim, (iBin / stride) % (data.edges[iDim].size() - 1));
   }
   return volume;
}

/// The scale factor of the whole sample, i.e. the product of the NormSys factors and the NormFactors.
double RooHistFactorySample::scaleFactor(double const *params) const
{
   Data const &data = *_data;
   return RooFit::Detail::MathFuncs::histFactoryScale(
      params, data.normSysParams.size(), data.normSysParams.data(), data.normSysCodes.data(), data.normSysLo.data(),
      data.normSysHi.data(), data.normFactorParams.size(), data.normFactorParams.data());
}

/// The value in a bin without the scale factor of the sample.
double RooHistFactorySample::binValue(double const *params, std::size_t iBin) const
{
   Data const &data = *_data;
   return RooFit::Detail::MathFuncs::histFactoryBin(
      params, iBin, data.numBins(), data.contents.data(), data.histoSysParams.size(), data.histoSysParams.data(),
      data.histoSysCodes.data(), data.histoSysLo.data(), data.histoSysHi.data(), data.binFactorNames.size(),
      data.binFactorParams.data());
}

template <class ParamValue>
void RooHistFactorySample::fillParams(ParamValue &&paramValue) const
{
   _paramVals.resize(_params.size());
   for (std::size_t i = 0; i < _params.size(); ++i) {
      _paramVals[i] = paramValue(static_cast<RooAbsReal const &>(_params[i]));
   }
}

/// The values of all bins at the current parameter values, either as densities or as yields.
void RooHistFactorySample::binValues(std::span<double> out, bool density) const
{
   if (out.size() != numBins()) {
      throw std::invalid_argument("RooHistFactorySample::binValues(): the output has the wrong size");
   }
   fillParams([](RooAbsReal const &param) { return param.getVal(); });
   const double coef = scaleFactor(_paramVals.data());
   for (std::size_t i = 0; i < out.size(); ++i) {
      double value = binValue(_paramVals.data(), i);
      if (density) {
         value *= 1. / binVolume(i);
      }
      out[i] = value * coef;
   }
}

std::size_t RooHistFactorySample::binIndex(std::span<const double> values) const
{
   Data const &data = *_data;
   std::size_t index = 0;
   for (std::size_t iDim = 0; iDim < data.edges.size(); ++iDim) {
      index = index * (data.edges[iDim].size() - 1) + binInDim(data, iDim, values[iDim]);
   }
   return index;
}

double RooHistFactorySample::evaluate() const
{
   fillParams([](RooAbsReal const &param) { return param.getVal(); });
   _obsVals.resize(_obs.size());
   for (std::size_t i = 0; i < _obs.size(); ++i) {
      _obsVals[i] = static_cast<RooAbsReal const &>(_obs[i]).getVal();
   }
   const std::size_t iBin = binIndex(_obsVals);
   double value = binValue(_paramVals.data(), iBin);
   if (!_yieldsMode) {
      value *= 1. / binVolume(iBin);
   }
   return value * scaleFactor(_paramVals.data());
}

void RooHistFactorySample::doEval(RooFit::EvalContext &ctx) const
{
   std::span<double> output = ctx.output();
   fillParams([&](RooAbsReal const &param) { return ctx.at(&param)[0]; });
   const double coef = scaleFactor(_paramVals.data());

   std::vector<std::span<const double>> obsValues;
   obsValues.reserve(_obs.size());
   for (RooAbsArg *obs : _obs) {
      obsValues.push_back(ctx.at(obs));
   }

   // For many entries, all bins are computed only once.
   const std::size_t nBins = numBins();
   const bool allBins = output.size() >= nBins;
   if (allBins) {
      _binVals.resize(nBins);
      for (std::size_t i = 0; i < nBins; ++i) {
         double value = binValue(_paramVals.data(), i);
         if (!_yieldsMode) {
            value *= 1. / binVolume(i);
         }
         _binVals[i] = value * coef;
      }
   }

   _obsVals.resize(_obs.size());
   for (std::size_t iEntry = 0; iEntry < output.size(); ++iEntry) {
      for (std::size_t i = 0; i < _obs.size(); ++i) {
         _obsVals[i] = obsValues[i].size() > 1 ? obsValues[i][iEntry] : obsValues[i][0];
      }
      const std::size_t iBin = binIndex(_obsVals);
      if (allBins) {
         output[iEntry] = _binVals[iBin];
         continue;
      }
      double value = binValue(_paramVals.data(), iBin);
      if (!_yieldsMode) {
         value *= 1. / binVolume(iBin);
      }
      output[iEntry] = value * coef;
   }
}

int RooHistFactorySample::observableIndex(RooAbsArg const &obs) const
{
   for (std::size_t i = 0; i < _obs.size(); ++i) {
      if (std::string(_obs[i].GetName()) == obs.GetName()) {
         return i;
      }
   }
   return -1;
}

bool RooHistFactorySample::isBinnedDistribution(const RooArgSet &obs) const
{
   for (RooAbsArg *arg : obs) {
      if (observableIndex(*arg) < 0) {
         return false;
      }
   }
   return true;
}

std::list<double> *RooHistFactorySample::binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   const int iDim = observableIndex(obs);
   if (iDim < 0) {
      return nullptr;
   }
   // Same as RooHistFunc::binBoundaries()
   const double delta = (xhi - xlo) * 1e-8;
   auto hint = new std::list<double>;
   for (double boundary : _data->edges[iDim]) {
      if (boundary > xlo - delta && boundary < xhi + delta) {
         hint->push_back(boundary);
      }
   }
   return hint;
}

std::list<double> *RooHistFactorySample::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   const int iDim = observableIndex(obs);
   if (iDim < 0) {
      return nullptr;
   }
   return RooCurve::plotSamplingHintForBinBoundaries(_data->edges[iDim], xlo, xhi);
}

/// Integrals over any subset of the observables are analytical. The code is one plus the bit mask of the integrated
/// observables.
Int_t RooHistFactorySample::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars,
                                                  const char * /*rangeName*/) const
{
   if (_forceNumInt || _yieldsMode) {
      return 0;
   }
   Int_t code = 0;
   for (std::size_t i = 0; i < _obs.size(); ++i) {
      if (RooAbsArg *arg = allVars.find(_obs[i])) {
         analVars.add(*arg);
         code |= 1 << i;
      }
   }
   return code == 0 ? 0 : code + 1;
}

double RooHistFactorySample::analyticalIntegral(Int_t code, const char *rangeName) const
{
   Data const &data = *_data;
   const std::vector<double> weights = integralWeights(code, rangeName);
   fillParams([](RooAbsReal const &param) { return param.getVal(); });
   const double integral = RooFit::Detail::MathFuncs::histFactoryIntegral(
      _paramVals.data(), weights.data(), data.numBins(), data.contents.data(), data.histoSysParams.size(),
      data.histoSysParams.data(), data.histoSysCodes.data(), data.histoSysLo.data(), data.histoSysHi.data(),
      data.binFactorNames.size(), data.binFactorParams.data());
   return integral * scaleFactor(_paramVals.data());
}

/// The weights of the bins in an integral with the code from getAnalyticalIntegral(). For an integrated observable, it
/// is the fraction of the bin in the integration range. For an observable that is not integrated, only the bin of the
/// current value has a weight, which is the inverse of its width because the integral is a density in this observable.
std::vector<double> RooHistFactorySample::integralWeights(Int_t code, const char *rangeName) const
{
   Data const &data = *_data;
   const std::size_t nDims = data.edges.size();
   const int mask = code - 1;

   std::vector<std::vector<double>> weightsPerDim(nDims);
   for (std::size_t iDim = 0; iDim < nDims; ++iDim) {
      const std::size_t nBinsDim = data.edges[iDim].size() - 1;
      auto const &obs = static_cast<RooRealVar const &>(_obs[iDim]);
      std::vector<double> &w = weightsPerDim[iDim];
      w.assign(nBinsDim, 0.0);
      if (!(mask & (1 << iDim))) {
         const std::size_t iBin = binInDim(data, iDim, obs.getVal());
         w[iBin] = 1. / binWidth(data, iDim, iBin);
         continue;
      }
      const double lo = obs.getMin(rangeName);
      const double hi = obs.getMax(rangeName);
      for (std::size_t iBin = 0; iBin < nBinsDim; ++iBin) {
         const double binLo = data.edges[iDim][iBin];
         const double binHi = data.edges[iDim][iBin + 1];
         if (binLo >= lo && binHi <= hi) {
            w[iBin] = 1.0;
         } else if (binHi > lo && binLo < hi) {
            w[iBin] = (std::min(hi, binHi) - std::max(lo, binLo)) / binWidth(data, iDim, iBin);
         }
      }
   }

   const std::size_t nBins = data.numBins();
   std::vector<double> weights(nBins);
   for (std::size_t iBin = 0; iBin < nBins; ++iBin) {
      double weight = 1.0;
      std::size_t stride = nBins;
      for (std::size_t iDim = 0; iDim < nDims; ++iDim) {
         const std::size_t nBinsDim = data.edges[iDim].size() - 1;
         stride /= nBinsDim;
         weight *= weightsPerDim[iDim][(iBin / stride) % nBinsDim];
      }
      weights[iBin] = weight;
   }
   return weights;
}

std::unique_ptr<RooAbsArg>
RooHistFactorySample::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   // In the binned likelihood, the values of the pdf are directly the yields
   // in the bins, like for a RooBinWidthFunction in that mode.
   if (ctx.binnedLikelihoodMode()) {
      auto newArg = std::make_unique<RooHistFactorySample>(*this);
      newArg->_yieldsMode = true;
      ctx.markAsCompiled(*newArg);
      ctx.compileServers(*newArg, {});
      ctx.setBinWidthFuncFlag(true);
      return newArg;
   }
   return RooAbsReal::compileForNormSet(normSet, ctx);
}

void RooHistFactorySample::setInterpCode(std::vector<int> Data::*codes, std::vector<int> Data::*params,
                                         RooAbsReal const &param, int code, int maxCode, const char *what)
{
   // By name, so that the parameter can also come from another copy of the model.
   const int iParam = _params.index(param.GetName());
   auto data = std::make_shared<Data>(*_data);
   bool found = false;
   for (std::size_t i = 0; i < (data.get()->*params).size(); ++i) {
      if ((data.get()->*params)[i] == iParam) {
         RooStats::HistFactory::Detail::setInterpolationCode(*this, what, param, data.get()->*codes, i, code, maxCode);
         found = true;
      }
   }
   if (!found) {
      coutE(InputArguments) << what << "::setInterpCode ERROR: " << param.GetName() << " is not a parameter of a "
                            << what << " modifier of sample \"" << sampleName() << "\"" << std::endl;
      return;
   }
   _data = std::move(data);
   setValueDirty();
}

/// Set the interpolation code of the HistoSys modifiers with the given parameter. The codes are the ones of
/// PiecewiseInterpolation.
void RooHistFactorySample::setHistoSysInterpCode(RooAbsReal const &param, int code)
{
   setInterpCode(&Data::histoSysCodes, &Data::histoSysParams, param, code, 6, "HistoSys");
}

/// Set the interpolation code of the NormSys modifiers with the given parameter. The codes are the ones of
/// FlexibleInterpVar.
void RooHistFactorySample::setNormSysInterpCode(RooAbsReal const &param, int code)
{
   setInterpCode(&Data::normSysCodes, &Data::normSysParams, param, code, 5, "NormSys");
}

void RooHistFactorySample::printMetaArgs(std::ostream &os) const
{
   os << "sample=" << sampleName() << " bins=" << numBins();
   auto printNames = [&](const char *label, std::vector<std::string> const &names) {
      if (names.empty()) {
         return;
      }
      os << " " << label << "=(";
      for (std::size_t i = 0; i < names.size(); ++i) {
         os << (i > 0 ? "," : "") << names[i];
      }
      os << ")";
   };
   printNames("normfactors", _data->normFactorNames);
   printNames("normsys", _data->normSysNames);
   printNames("histosys", _data->histoSysNames);
   printNames("binfactors", _data->binFactorNames);
   os << " ";
}

void RooHistFactorySample::Streamer(TBuffer &buffer)
{
   static TClass *dataClass = TClass::GetClass<Data>();
   if (buffer.IsReading()) {
      buffer.ReadClassBuffer(RooHistFactorySample::Class(), this);
      auto data = std::make_shared<Data>();
      buffer.StreamObject(data.get(), dataClass);
      _data = std::move(data);
   } else {
      buffer.WriteClassBuffer(RooHistFactorySample::Class(), this);
      buffer.StreamObject(const_cast<Data *>(_data.get()), dataClass);
   }
}
