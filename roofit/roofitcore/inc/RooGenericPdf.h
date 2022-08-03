/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGenericPdf.h,v 1.20 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_GENERIC_PDF
#define ROO_GENERIC_PDF

#include "RooAbsPdf.h"
#include "RooFormula.h"
#include "RooListProxy.h"

class RooArgList ;

class RooGenericPdf : public RooAbsPdf {
public:
  // Constructors, assignment etc
  inline RooGenericPdf(){}
  RooGenericPdf(const char *name, const char *title, const char* formula, const RooArgList& dependents);
  RooGenericPdf(const char *name, const char *title, const RooArgList& dependents);
  RooGenericPdf(const RooGenericPdf& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooGenericPdf(*this,newname); }

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override ;
  void printMetaArgs(std::ostream& os) const override ;

  // Debugging
  void dumpFormula() { formula().dump() ; }

  const char* expression() const { return _formExpr.Data(); }
  const RooArgList& dependents() const { return _actualVars; }

protected:

  RooFormula& formula() const ;

  // Function evaluation
  RooListProxy _actualVars ;
  double evaluate() const override ;
  RooSpan<double> evaluateSpan(RooBatchCompute::RunContext& inputData, const RooArgSet* normSet) const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;

  bool setFormula(const char* formula) ;

  // Post-processing of server redirection
  bool redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive) override ;

  bool isValidReal(double value, bool printError) const override ;

  std::unique_ptr<RooFormula> _formula{nullptr}; ///<! Formula engine
  TString _formExpr ;            ///< Formula expression string

  ClassDefOverride(RooGenericPdf,1) // Generic PDF defined by string expression and list of variables
};

#endif
