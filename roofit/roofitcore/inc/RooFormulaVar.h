/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooFormulaVar.h,v 1.29 2007/08/09 19:55:47 wouter Exp $
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
#ifndef ROO_FORMULA_VAR
#define ROO_FORMULA_VAR

#include "RooAbsReal.h"
#include "RooFormula.h"
#include "RooArgList.h"
#include "RooListProxy.h"
#include "RooTrace.h"

#include <memory>
#include <list>

class RooArgSet ;

class RooFormulaVar : public RooAbsReal {
public:
  // Constructors, assignment etc
  RooFormulaVar() { }
  RooFormulaVar(const char *name, const char *title, const char* formula, const RooArgList& dependents, bool checkVariables = true);
  RooFormulaVar(const char *name, const char *title, const RooArgList& dependents, bool checkVariables = true);
  RooFormulaVar(const RooFormulaVar& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooFormulaVar(*this,newname); }

  inline bool ok() const { return getFormula().ok() ; }
  const char* expression() const { return _formExpr.Data(); }
  const RooArgList& dependents() const { return _actualVars; }

  /// Return pointer to parameter with given name.
  inline RooAbsArg* getParameter(const char* name) const {
    return _actualVars.find(name) ;
  }
  /// Return pointer to parameter at given index.
  inline RooAbsArg* getParameter(Int_t index) const {
    return _actualVars.at(index) ;
  }

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent= "") const override ;
  void printMetaArgs(std::ostream& os) const override ;

  // Debugging
  /// Dump the formula to stdout.
  void dumpFormula() { getFormula().dump() ; }
  /// Get reference to the internal formula object.
  const RooFormula& formula() const {
    return getFormula();
  }

  Double_t defaultErrorLevel() const override ;

  std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override ;
  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override ;

  // Function evaluation
  Double_t evaluate() const override ;
  RooSpan<double> evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const override;
  inline void computeBatch(cudaStream_t* stream, double* output, size_t nEvents, RooBatchCompute::DataMap& dataMap) const override
  {
    formula().computeBatch(stream, output, nEvents, dataMap);
  }


  protected:
  // Post-processing of server redirection
  bool redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive) override ;

  bool isValidReal(Double_t /*value*/, bool /*printError*/) const override {return true;}

  private:
  RooFormula& getFormula() const;

  RooListProxy _actualVars ;     ///< Actual parameters used by formula engine
  std::unique_ptr<RooFormula> _formula{nullptr}; ///<! Formula engine
  mutable RooArgSet* _nset{nullptr}; ///<! Normalization set to be passed along to contents
  TString _formExpr ;            ///< Formula expression string

  ClassDefOverride(RooFormulaVar,1) // Real-valued function of other RooAbsArgs calculated by a TFormula expression
};

#endif
