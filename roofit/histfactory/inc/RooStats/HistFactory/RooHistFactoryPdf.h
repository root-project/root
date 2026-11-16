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

#ifndef RooStats_HistFactory_RooHistFactoryPdf_h
#define RooStats_HistFactory_RooHistFactoryPdf_h

#include <RooRealSumPdf.h>

#include <string>

class RooHistFactorySample;

class RooHistFactoryPdf : public RooRealSumPdf {
public:
   RooHistFactoryPdf() = default;
   RooHistFactoryPdf(const char *name, const char *title, RooArgList const &samples);
   RooHistFactoryPdf(const RooHistFactoryPdf &other, const char *name = nullptr);
   TObject *clone(const char *newname = nullptr) const override { return new RooHistFactoryPdf(*this, newname); }

   std::size_t numSamples() const { return _funcList.size(); }
   RooHistFactorySample const &sample(std::size_t i) const;
   RooHistFactorySample const *sample(std::string const &sampleName) const;
   RooArgList const &observables() const;

   void printMetaArgs(std::ostream &os) const override;

   ClassDefOverride(RooHistFactoryPdf, 1);
};

#endif
