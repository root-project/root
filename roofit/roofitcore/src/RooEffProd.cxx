/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, NIKHEF
 *   GR, Gerhard Raven, NIKHEF/VU                                            *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


/////////////////////////////////////////////////////////////////////////////////////
/// \class RooEffProd
/// The class RooEffProd implements the product of a PDF with an efficiency function.
/// The normalization integral of the product is calculated numerically, but the
/// event generation is handled by a specialized generator context that implements
/// the event generation in a more efficient for cases where the PDF has an internal
/// generator that is smarter than accept reject.
///

#include "RooFit.h"
#include "RooEffProd.h"
#include "RooEffGenContext.h"
#include "RooNameReg.h"
#include "RooRealVar.h"


////////////////////////////////////////////////////////////////////////////////
/// Constructor of a a production of p.d.f inPdf with efficiency
/// function inEff.

RooEffProd::RooEffProd(const char *name, const char *title,
                             RooAbsPdf& inPdf, RooAbsReal& inEff) :
  RooAbsPdf(name,title),
  _pdf("pdf","pre-efficiency pdf", this,inPdf),
  _eff("eff","efficiency function",this,inEff)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooEffProd::RooEffProd(const RooEffProd& other, const char* name) :
  RooAbsPdf(other, name),
  _pdf("pdf",this,other._pdf),
  _eff("acc",this,other._eff)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return 'raw' unnormalized value of p.d.f

Double_t RooEffProd::evaluate() const
{
  return _eff * _pdf;
}


////////////////////////////////////////////////////////////////////////////////
/// Return specialized generator context for RooEffProds that implements generation
/// in a more efficient way than can be done for generic correlated products

RooAbsGenContext* RooEffProd::genContext(const RooArgSet &vars, const RooDataSet *prototype,
                                            const RooArgSet* auxProto, Bool_t verbose) const
{
  return new RooEffGenContext(*this,
                              static_cast<RooAbsPdf const&>(_pdf.arg()),
                              static_cast<RooAbsReal const&>(_eff.arg()),
                              vars,prototype,auxProto,verbose) ;
}
