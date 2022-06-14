/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealVarSharedProperties.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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

/**
\file RooRealVarSharedProperties.h
\class RooRealVarSharedProperties
\ingroup Roofitcore

Class RooRealVarSharedProperties is an implementation of RooSharedProperties
that stores the properties of a RooRealVar that are shared among clones.
For RooRealVars these are the definitions of the named ranges.
**/

#ifndef ROO_REAL_VAR_SHARED_PROPERTY
#define ROO_REAL_VAR_SHARED_PROPERTY

#include "RooAbsBinning.h"
#include "RooSharedProperties.h"

#include <memory>
#include <unordered_map>
#include <string>

class RooAbsBinning;

class RooRealVarSharedProperties : public RooSharedProperties {
public:

  /// Default constructor.
  RooRealVarSharedProperties() {}
  /// Constructor with unique-id string.
  RooRealVarSharedProperties(const char* uuidstr) : RooSharedProperties(uuidstr) {}

  /// Destructor
  ~RooRealVarSharedProperties() override {
    if (_ownBinnings) {
      for (auto& item : _altBinning) {
        delete item.second;
      }
    }
  }

  void disownBinnings() {
    _ownBinnings = false;
  }

protected:

  friend class RooRealVar ;

  std::unordered_map<std::string,RooAbsBinning*> _altBinning ;  ///< Optional alternative ranges and binnings
  bool _ownBinnings{true}; //!
  ClassDefOverride(RooRealVarSharedProperties,2) // Shared properties of a RooRealVar clone set
};


#endif
