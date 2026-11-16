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

/** \class RooHistFactoryPdf
\ingroup HistFactory

A HistFactory channel: the sum of its samples, which are RooHistFactorySample objects. It corresponds to the
`histfactory_dist` of the HS3 standard, without the constraint terms of the parameters, which are separate pdfs.

The pdf is a RooRealSumPdf whose coefficients are all one, so it supports the binned likelihood optimization
(attribute `BinnedLikelihood`), binned generation (attribute `GenerateBinned`), and the plotting of single samples
with `RooFit::Components()`. The samples share the observables and their binning.
**/

#include <RooStats/HistFactory/RooHistFactoryPdf.h>

#include <RooStats/HistFactory/RooHistFactorySample.h>

#include <RooConstVar.h>
#include <RooGlobalFunc.h>

#include <stdexcept>

namespace {

RooArgList unitCoefficients(std::size_t n)
{
   RooArgList coefs;
   for (std::size_t i = 0; i < n; ++i) {
      coefs.add(RooFit::RooConst(1.0));
   }
   return coefs;
}

void checkSamples(const char *name, RooArgList const &samples)
{
   if (samples.empty()) {
      throw std::invalid_argument(std::string("RooHistFactoryPdf \"") + name + "\": there are no samples");
   }
   RooHistFactorySample const *first = nullptr;
   for (RooAbsArg *arg : samples) {
      auto *sample = dynamic_cast<RooHistFactorySample const *>(arg);
      if (!sample) {
         throw std::invalid_argument(std::string("RooHistFactoryPdf \"") + name + "\": \"" + arg->GetName() +
                                     "\" is not a RooHistFactorySample");
      }
      if (!first) {
         first = sample;
         continue;
      }
      bool same =
         sample->observables().size() == first->observables().size() && sample->data().edges == first->data().edges;
      for (std::size_t i = 0; same && i < first->observables().size(); ++i) {
         same = &sample->observables()[i] == &first->observables()[i];
      }
      if (!same) {
         throw std::invalid_argument(std::string("RooHistFactoryPdf \"") + name + "\": the sample \"" +
                                     sample->GetName() + "\" has other observables or bins than the sample \"" +
                                     first->GetName() + "\"");
      }
   }
}

} // namespace

/// Create a HistFactory channel from its samples, which need to have the same observables and bins.
RooHistFactoryPdf::RooHistFactoryPdf(const char *name, const char *title, RooArgList const &samples)
   : RooRealSumPdf(name, title, (checkSamples(name, samples), samples), unitCoefficients(samples.size()), true)
{
}

RooHistFactoryPdf::RooHistFactoryPdf(const RooHistFactoryPdf &other, const char *name) : RooRealSumPdf(other, name) {}

RooHistFactorySample const &RooHistFactoryPdf::sample(std::size_t i) const
{
   return static_cast<RooHistFactorySample const &>(_funcList[i]);
}

/// The sample with a given name in the HistFactory model, or nullptr if there is none.
RooHistFactorySample const *RooHistFactoryPdf::sample(std::string const &sampleName) const
{
   for (std::size_t i = 0; i < numSamples(); ++i) {
      if (sample(i).sampleName() == sampleName) {
         return &sample(i);
      }
   }
   return nullptr;
}

RooArgList const &RooHistFactoryPdf::observables() const
{
   return sample(0).observables();
}

void RooHistFactoryPdf::printMetaArgs(std::ostream &os) const
{
   os << "samples=(";
   for (std::size_t i = 0; i < numSamples(); ++i) {
      os << (i > 0 ? "," : "") << sample(i).GetName();
   }
   os << ") ";
}
