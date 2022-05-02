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
#ifndef ROO_CONSTRAINT_SUM
#define ROO_CONSTRAINT_SUM

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooSetProxy.h"

class RooRealVar;
class RooArgList ;
class RooWorkspace ;

class RooConstraintSum : public RooAbsReal {
public:

  RooConstraintSum() {}
  RooConstraintSum(const char *name, const char *title, const RooArgSet& constraintSet, const RooArgSet& paramSet, bool takeGlobalObservablesFromData=false) ;

  RooConstraintSum(const RooConstraintSum& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const override { return new RooConstraintSum(*this, newname); }

  const RooArgList& list() { return _set1 ; }

  static std::unique_ptr<RooAbsReal> createConstraintTerm(
        std::string const& name,
        RooAbsPdf const& pdf,
        RooAbsData const& data,
        RooArgSet const* constrainedParameters,
        RooArgSet const* externalConstraints,
        RooArgSet const* globalObservables,
        const char* globalObservablesTag,
        bool takeGlobalObservablesFromData,
        bool cloneConstraints,
        RooWorkspace * workspace);

  bool setData(RooAbsData const& data, bool cloneData=true);
  /// \copydoc setData(RooAbsData const&, bool)
  bool setData(RooAbsData& data, bool cloneData=true) override {
    return setData(static_cast<RooAbsData const&>(data), cloneData);
  }

  void fillNormSetForServer(RooArgSet const& /*normSet*/, RooAbsArg const& server, RooArgSet& serverNormSet) const override {
    for(auto * arg : _paramSet) if(server.dependsOn(*arg)) serverNormSet.add(*arg);
  }

protected:

  RooListProxy _set1 ;    // Set of constraint terms
  RooSetProxy _paramSet ; // Set of parameters to which constraints apply
  const bool _takeGlobalObservablesFromData = false; // If the global observable values are taken from data

  Double_t evaluate() const override;

  ClassDefOverride(RooConstraintSum,3) // sum of -log of set of RooAbsPdf representing parameter constraints
};

#endif
