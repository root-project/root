/// \cond ROOFIT_INTERNAL

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_ABS_NUM_GENERATOR
#define ROO_ABS_NUM_GENERATOR

#include "RooArgSet.h"
#include "RooArgList.h"

class RooAbsReal;
class RooRealVar;
class RooDataSet;
class RooRealBinding;
class RooNumGenConfig ;

class RooAbsNumGenerator {
public:
  RooAbsNumGenerator();
  RooAbsNumGenerator(const RooAbsReal &func, const RooArgSet &genVars, bool verbose=false, const RooAbsReal* maxFuncVal=nullptr);
  virtual RooAbsNumGenerator* clone(const RooAbsReal&, const RooArgSet& genVars, const RooArgSet& condVars,
                const RooNumGenConfig& config, bool verbose=false, const RooAbsReal* maxFuncVal=nullptr) const = 0 ;

  bool isValid() const {
    // If true, generator is in a valid state
    return _isValid;
  }
  virtual ~RooAbsNumGenerator() ;

  inline void setVerbose(bool verbose= true) {
    // If flag is true, verbose messaging will be active during generation
    _verbose= verbose;
  }
  inline bool isVerbose() const {
    // Return status of verbose messaging flag
    return _verbose;
  }

  virtual const RooArgSet *generateEvent(UInt_t remaining, double& resampleRatio) = 0;
  virtual double getFuncMax() { return 0 ; }

  void attachParameters(const RooArgSet& vars) ;

  // Advertisement of capabilities
  virtual bool canSampleCategories() const { return false ; }
  virtual bool canSampleConditional() const { return false ; } // Must implement getFuncMax()

  /// Return unique name of generator implementation
  virtual std::string const& generatorName() const = 0;

protected:

   RooArgSet _cloneSet;                      ///< Set owning clone of input function
   RooAbsReal *_funcClone = nullptr;         ///< Pointer to top level node of cloned function
   const RooAbsReal *_funcMaxVal = nullptr;  ///< Container for maximum function value
   RooArgSet _catVars;                       ///< Set of discrete observabeles
   RooArgSet _realVars;                      ///< Set of real valued observabeles
   bool _verbose = false;                    ///< Verbose flag
   bool _isValid = false;                    ///< Valid flag
   std::unique_ptr<RooAbsArg> _funcValStore; ///< RRV storing function value in context
   RooRealVar *_funcValPtr = nullptr;        ///< RRV storing function value in output dataset
   std::unique_ptr<RooDataSet> _cache;       ///< Dataset holding generared values of observables
};

#endif

/// \endcond
