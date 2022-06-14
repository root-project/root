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

#ifndef ROOFUNCTORBINDING
#define ROOFUNCTORBINDING

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooListProxy.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooMsgService.h"
#include "Math/IFunction.h"

namespace RooFit {

RooAbsReal* bindFunction(const char* name, const ROOT::Math::IBaseFunctionMultiDim& ftor,const RooArgList& vars) ;
RooAbsPdf*  bindPdf(const char* name, const ROOT::Math::IBaseFunctionMultiDim& ftor, const RooArgList& vars) ;

}

class RooFunctorBinding : public RooAbsReal {
public:
  RooFunctorBinding() : func(0), x(0) {
    // Default constructor
  } ;
  RooFunctorBinding(const char *name, const char *title, const ROOT::Math::IBaseFunctionMultiDim& ftor, const RooArgList& vars);
  RooFunctorBinding(const RooFunctorBinding& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooFunctorBinding(*this,newname); }
  inline ~RooFunctorBinding() override { delete[] x ; }
  void printArgs(std::ostream& os) const override ;

protected:

  double evaluate() const override ;

  const ROOT::Math::IBaseFunctionMultiDim* func ;    // Functor
  RooListProxy                       vars ;    // Argument reference
  double*                           x ; // Argument value array


private:

  ClassDefOverride(RooFunctorBinding,1) // RooAbsReal binding to a ROOT::Math::IBaseFunctionMultiDim
};



class RooFunctorPdfBinding : public RooAbsPdf {
public:
  RooFunctorPdfBinding() : func(0), x(0) {
    // Default constructor
  } ;
  RooFunctorPdfBinding(const char *name, const char *title, const ROOT::Math::IBaseFunctionMultiDim& ftor, const RooArgList& vars);
  RooFunctorPdfBinding(const RooFunctorPdfBinding& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooFunctorPdfBinding(*this,newname); }
  inline ~RooFunctorPdfBinding() override { delete[] x ; }
  void printArgs(std::ostream& os) const override ;

protected:

  double evaluate() const override ;

  const ROOT::Math::IBaseFunctionMultiDim* func ;    // Functor
  RooListProxy                       vars ;    // Argument reference
  double*                             x ; // Argument value array


private:

  ClassDefOverride(RooFunctorPdfBinding,1) // RooAbsPdf binding to a ROOT::Math::IBaseFunctionMultiDim
};


#endif
