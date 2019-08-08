// Author: Stephan Hageboeck, CERN  25 Feb 2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "BatchHelpers.h"

namespace BatchHelpers {

/* This function returns the minimum size of the non-zero-sized batches
 * It is used when the number of parameters are<=3 and explicit instantiation
 * will be used. 
 */
size_t findSize(std::vector< RooSpan<const double> > parameters) 
{
  size_t ret = SIZE_MAX;
  for (auto &param : parameters) 
    if (param.size()> 0 && param.size()<ret) ret=param.size();
    
  return ret;
}

/* This function returns the minimum size of the non-zero-sized batches
 * as well as the number of parameters that are batches, wrapped in a
 * EvaluateInfo struct (see BatchHelpers.h). It will be used when the 
 * number of parameters is > 3 and the BracketAdapterWithMask will be used.
 */
EvaluateInfo getInfo(std::vector<const RooRealProxy*> parameters, size_t begin, size_t batchSize)
{
  EvaluateInfo ret = {SIZE_MAX, 0};
  for (const auto par : parameters) {
    RooSpan<const double> span = par->getValBatch(begin,batchSize);
    if ( !span.empty() ) {
      ret.nBatches++;
      if (span.size() < ret.size) ret.size = span.size();
    }
  }
  return ret;
}

};
