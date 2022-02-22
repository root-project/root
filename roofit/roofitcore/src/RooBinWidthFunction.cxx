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
 *
 * RooBinWidthFunction is a class that returns the bin width (or volume) given a RooHistFunc.
 * It can be used to normalise by bin width or to compute event densities. Using the extra
 * argument of the constructor, it can also return the inverse of the bin width (or volume).
 */

#include "RooBinWidthFunction.h"

#include "RooDataHist.h"
#include "RunContext.h"


/// Compute current bin of observable, and return its volume or inverse volume, depending
/// on configuration chosen in the constructor.
/// If the bin is not valid, return a volume of 1.
double RooBinWidthFunction::evaluate() const {
  const RooDataHist& dataHist = _histFunc->dataHist();
  const auto idx = _histFunc->getBin();
  auto volumes = dataHist.binVolumes(0, dataHist.numEntries());
  const double volume = idx >= 0 ? volumes[idx] : 1.;

  return _divideByBinWidth ? 1./volume : volume;
}


/// Compute bin index for all values of the observable(s) in `evalData`, and return their volumes or inverse volumes, depending
/// on the configuration chosen in the constructor.
/// If a bin is not valid, return a volume of 1.
void RooBinWidthFunction::computeBatch(cudaStream_t*, double* output, size_t, RooBatchCompute::DataMap& dataMap) const {
  const RooDataHist& dataHist = _histFunc->dataHist();
  std::vector<Int_t> bins = _histFunc->getBins(dataMap);
  auto volumes = dataHist.binVolumes(0, dataHist.numEntries());

  if (_divideByBinWidth) {
    for (std::size_t i=0; i < bins.size(); ++i) {
      output[i] = bins[i] >= 0 ? 1./volumes[bins[i]] : 1.;
    }
  } else {
    for (std::size_t i=0; i < bins.size(); ++i) {
      output[i] = bins[i] >= 0 ? volumes[bins[i]] : 1.;
    }
  }
}
