/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooEffProd.cc,v 1.2 2005/06/23 07:37:30 wverkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven, NIKHEF/VU                                            *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


// -- CLASS DESCRIPTION [PDF] --
// The class RooEffProd implements the product of a PDF with an efficiency function.
// The normalization integral of the product is calculated numerically, but the
// event generation is handled by a specialized generator context that implements
// the event generation in a more efficient for cases where the PDF has an internal
// generator that is smarter than accept reject. 

#include "RooFit.h"
#include "RooEffProd.h"
#include "RooEffGenContext.h"

ClassImp(RooEffProd)
  ;

RooEffProd::RooEffProd(const char *name, const char *title, 
                             RooAbsPdf& pdf, RooAbsReal& eff) :
  RooAbsPdf(name,title),
  _pdf("pdf","pre-efficiency pdf", this,pdf),
  _eff("eff","efficiency function",this,eff)
{  

}


RooEffProd::RooEffProd(const RooEffProd& other, const char* name) : 
  RooAbsPdf(other, name),
  _pdf("pdf",this,other._pdf),
  _eff("acc",this,other._eff)
{
}


RooEffProd::~RooEffProd() 
{
}

Double_t RooEffProd::evaluate() const
{
    return eff()->getVal() * pdf()->getVal();
}

RooAbsGenContext* RooEffProd::genContext(const RooArgSet &vars, const RooDataSet *prototype,
                                            const RooArgSet* auxProto, Bool_t verbose) const
{
  assert(pdf()!=0);
  assert(eff()!=0);
  return new RooEffGenContext(*this,*pdf(),*eff(),vars,prototype,auxProto,verbose) ;
}
