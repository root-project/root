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
#include "RooListProxy.h"
#include "RooAbsBinning.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

class RooArgList ;
class RooFormulaEvaluator;
class RooAbsRealLValue;

class RooGenericPdf : public RooAbsPdf {
public:
  // Constructors, assignment etc
  RooGenericPdf();
  ~RooGenericPdf() override;
  RooGenericPdf(const char *name, const char *title, const char* formula, const RooArgList& dependents);
  RooGenericPdf(const char *name, const char *title, const RooArgList& dependents);
  RooGenericPdf(const RooGenericPdf& other, const char* name=nullptr);
  TObject* clone(const char* newname=nullptr) const override { return new RooGenericPdf(*this,newname); }

  bool canComputeBatchWithCuda() const override;

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  /// Return pointer to parameter with given name.
  inline RooAbsArg* getParameter(const char* name) const {
    return _actualVars.find(name) ;
  }
  /// Return pointer to parameter at given index.
  inline RooAbsArg* getParameter(Int_t index) const {
    return _actualVars.at(index) ;
  }
  /// Return the number of parameters.
  inline size_t nParameters() const {
    return _actualVars.size();
  }

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override ;
  void printMetaArgs(std::ostream& os) const override ;

  // Debugging
  void dumpFormula();

  const char* expression() const { return _formExpr.Data(); }
  const RooArgList& dependents() const { return _actualVars; }

  /// Name of the cling-JIT-compiled function that evaluates this formula,
  /// which generated code from the codegen fallback path calls by name.
  /// \note Returns an empty string when the formula is handled by the JIT-free
  /// formula backend (the default for supported expressions, see
  /// formulaUsesAstBackend()); codegen then inlines the expression via
  /// emitFormulaCpp() instead.
  std::string getUniqueFuncName() const;
  std::string emitFormulaCpp(std::function<std::string(unsigned int)> const &varName) const;
  bool formulaUsesAstBackend() const;

  void setBinning(const RooAbsRealLValue &obs, const RooAbsBinning &binning, bool checkFlatness = true);
  const RooAbsBinning *getBinning(const RooAbsRealLValue &obs) const;
  bool removeBinning(const RooAbsRealLValue &obs);

  bool isBinnedDistribution(const RooArgSet &obs) const override;
  std::list<double> *binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const override;
  std::list<double> *plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const override;

protected:
   RooFormulaEvaluator &evaluator() const;

   // Function evaluation
   RooListProxy _actualVars;
   double evaluate() const override;
   void doEval(RooFit::EvalContext &) const override;

   bool isValidReal(double /*value*/, bool /*printError*/) const override { return true; }

   mutable std::unique_ptr<RooFormulaEvaluator> _evaluator; ///<! Formula evaluation engine
   TString _formExpr;                                       ///< Formula expression string

   std::map<int, std::unique_ptr<RooAbsBinning>> _binnings; ///< User-defined binnings, keyed by the observable's index
                                                            ///< in _actualVars, for a piecewise-flat distribution

   ClassDefOverride(RooGenericPdf, 2) // Generic PDF defined by string expression and list of variables
};

#endif
