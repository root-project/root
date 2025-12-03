/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOTFNPDFBINDING
#define ROOTFNPDFBINDING

#include "RooListProxy.h"
#include "RooAbsPdf.h"
class TF1 ;
class TF2 ;
class TF3 ;

class RooTFnPdfBinding : public RooAbsPdf {
public:
  RooTFnPdfBinding() = default;
  RooTFnPdfBinding(const char *name, const char *title, TF1* func, const RooArgList& list);
  RooTFnPdfBinding(const RooTFnPdfBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooTFnPdfBinding(*this,newname); }

  void printArgs(std::ostream& os) const override ;

protected:

  RooListProxy _list ;
  TF1* _func = nullptr;

  double evaluate() const override ;

private:

  ClassDefOverride(RooTFnPdfBinding,1) // RooAbsPdf binding to ROOT TF[123] functions
};


namespace RooFit {

RooAbsPdf* bindPdf(TF1* func,RooAbsReal& x) ;
RooAbsPdf* bindPdf(TF2* func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsPdf* bindPdf(TF3* func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;

}


#endif
