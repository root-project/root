/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, NIKHEF, verkerke@nikhef.nl                         *
 *                                                                           *
 * Copyright (c) 2000-2008, NIKHEF, Regents of the University of California  *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 *****************************************************************************/

#ifndef ROOFUNCTOR1DBINDING
#define ROOFUNCTOR1DBINDING

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooListProxy.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooMsgService.h"
#include "Math/IFunction.h"


namespace RooFit {

RooAbsReal* bindFunction(const char* name, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& vars) ;
RooAbsPdf*  bindPdf(const char* name, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& vars) ;

}


class RooFunctor1DBinding : public RooAbsReal {
public:
  RooFunctor1DBinding() : func(nullptr) {
    // Default constructor
  } ;
  RooFunctor1DBinding(const char *name, const char *title, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& var);
  RooFunctor1DBinding(const RooFunctor1DBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooFunctor1DBinding(*this,newname); }
  inline ~RooFunctor1DBinding() override {}
  void printArgs(std::ostream& os) const override ;

protected:

  double evaluate() const override ;

  const ROOT::Math::IBaseFunctionOneDim* func ;    // Functor
  RooRealProxy                       var ;    // Argument reference


private:

  ClassDefOverride(RooFunctor1DBinding,1) // RooAbsReal binding to a ROOT::Math::IBaseFunctionOneDim
};



class RooFunctor1DPdfBinding : public RooAbsPdf {
public:
  RooFunctor1DPdfBinding() : func(nullptr) {
    // Default constructor
  } ;
  RooFunctor1DPdfBinding(const char *name, const char *title, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& vars);
  RooFunctor1DPdfBinding(const RooFunctor1DPdfBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooFunctor1DPdfBinding(*this,newname); }
  inline ~RooFunctor1DPdfBinding() override {}
  void printArgs(std::ostream& os) const override ;

protected:

  double evaluate() const override ;

  const ROOT::Math::IBaseFunctionOneDim* func ;    // Functor
  RooRealProxy                           var ;    // Argument reference


private:

  ClassDefOverride(RooFunctor1DPdfBinding,1) // RooAbsPdf binding to a ROOT::Math::IBaseFunctionOneDim
};


#endif
