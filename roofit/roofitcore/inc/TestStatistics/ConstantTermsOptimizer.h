// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOT_ROOFIT_TESTSTATISTICS_ConstantTermsOptimizer
#define ROOT_ROOFIT_TESTSTATISTICS_ConstantTermsOptimizer

// forward declarations
class RooAbsReal;
class RooArgSet;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

// this is a class only for convenience: it saves multiple friend definitions in RooAbsData for otherwise free functions
struct ConstantTermsOptimizer {
   static void enableConstantTermsOptimization(RooAbsReal *function, RooArgSet *norm_set, RooAbsData *dataset,
                                           bool applyTrackingOpt);
   static void optimizeCaching(RooAbsReal *function, RooArgSet *norm_set, RooArgSet* observables, RooAbsData *dataset);
   static void disableConstantTermsOptimization(RooAbsReal *function, RooArgSet *norm_set, RooArgSet* observables, RooAbsData *dataset);
   static RooArgSet requiredExtraObservables();
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_ConstantTermsOptimizer
