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
  RooRealBinding(const RooAbsReal& func, const RooArgSet &vars, const RooArgSet* nset=nullptr, bool clipInvalid=false, const TNamed* rangeName=nullptr);
  RooRealBinding(const RooRealBinding& other, const RooArgSet* nset=nullptr) ;
  ~RooRealBinding() override;

  double operator()(const double xvector[]) const override;
  virtual RooSpan<const double> getValues(std::vector<RooSpan<const double>> coordinates) const;
  RooSpan<const double> getValuesOfBoundFunction(RooBatchCompute::RunContext& evalData) const;
  double getMinLimit(UInt_t dimension) const override;
  double getMaxLimit(UInt_t dimension) const override;

  void saveXVec() const override ;
  void restoreXVec() const override ;

  const char* getName() const override ;

  std::list<double>* binBoundaries(Int_t) const override ;
  /// Return a pointer to the observable that defines the `i`-th dimension of the function.
  RooAbsRealLValue* observable(unsigned int i) const { return i < _vars.size() ? _vars[i] : nullptr; }
  std::list<double>* plotSamplingHint(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;

protected:

  void loadValues(const double xvector[]) const;
  const RooAbsReal *_func;
  std::vector<RooAbsRealLValue*> _vars; ///< Non-owned pointers to variables
  const RooArgSet *_nset;
  mutable bool _xvecValid;
  bool _clipInvalid ;
  mutable double* _xsave ;
  const TNamed* _rangeName ; ///<!

  mutable std::vector<RooAbsReal*> _compList ; ///<!
  mutable std::vector<double>    _compSave ; ///<!
  mutable double _funcSave ; ///<!
  mutable std::unique_ptr<RooBatchCompute::RunContext> _evalData; ///< Memory for batch evaluations

  ClassDefOverride(RooRealBinding,0) // Function binding to RooAbsReal object
};

#endif

