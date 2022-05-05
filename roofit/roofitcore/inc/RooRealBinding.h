/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealBinding.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_BINDING
#define ROO_REAL_BINDING

#include "RooAbsFunc.h"
#include "RooSpan.h"
#include <vector>
#include <memory>

class RooAbsRealLValue;
class RooAbsReal;
class RooArgSet;
namespace RooBatchCompute{ struct RunContext; }

class RooRealBinding : public RooAbsFunc {
public:
  RooRealBinding(const RooAbsReal& func, const RooArgSet &vars, const RooArgSet* nset=0, bool clipInvalid=false, const TNamed* rangeName=0);
  RooRealBinding(const RooRealBinding& other, const RooArgSet* nset=0) ;
  ~RooRealBinding() override;

  Double_t operator()(const Double_t xvector[]) const override;
  virtual RooSpan<const double> getValues(std::vector<RooSpan<const double>> coordinates) const;
  RooSpan<const double> getValuesOfBoundFunction(RooBatchCompute::RunContext& evalData) const;
  Double_t getMinLimit(UInt_t dimension) const override;
  Double_t getMaxLimit(UInt_t dimension) const override;

  void saveXVec() const override ;
  void restoreXVec() const override ;

  const char* getName() const override ;

  std::list<Double_t>* binBoundaries(Int_t) const override ;
  /// Return a pointer to the observable that defines the `i`-th dimension of the function.
  RooAbsRealLValue* observable(unsigned int i) const { return i < _vars.size() ? _vars[i] : nullptr; }
  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override ;

protected:

  void loadValues(const Double_t xvector[]) const;
  const RooAbsReal *_func;
  std::vector<RooAbsRealLValue*> _vars; ///< Non-owned pointers to variables
  const RooArgSet *_nset;
  mutable bool _xvecValid;
  bool _clipInvalid ;
  mutable Double_t* _xsave ;
  const TNamed* _rangeName ; ///<!

  mutable std::vector<RooAbsReal*> _compList ; ///<!
  mutable std::vector<Double_t>    _compSave ; ///<!
  mutable Double_t _funcSave ; ///<!
  mutable std::unique_ptr<RooBatchCompute::RunContext> _evalData; ///< Memory for batch evaluations

  ClassDefOverride(RooRealBinding,0) // Function binding to RooAbsReal object
};

#endif

