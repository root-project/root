/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooBreitWigner.cc,v 1.1 2001/09/14 23:48:11 schieti Exp $
 * Authors:
 *   AS, Abi Soffer, Colorado State University, abi@slac.stanford.edu
 *   TS, Thomas Schietinger, SLAC, schieti@slac.stanford.edu
 * History:
 *   13-Mar-2001 AS Created.
 *   14-Sep-2001 TS Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 2001 Colorado State University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooBreitWigner.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
// #include "RooFitTools/RooRandom.hh"

ClassImp(RooBreitWigner)

RooBreitWigner::RooBreitWigner(const char *name, const char *title,
			 RooAbsReal& _x, RooAbsReal& _mean,
			 RooAbsReal& _width) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  width("width","Width",this,_width)
{
}


RooBreitWigner::RooBreitWigner(const RooBreitWigner& other, const char* name) : 
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  width("width",this,other.width)
{
}


Double_t RooBreitWigner::evaluate() const
{
  Double_t arg= x - mean;  
  return 1. / (arg*arg + 0.25*width*width);
}


Int_t RooBreitWigner::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}


Double_t RooBreitWigner::analyticalIntegral(Int_t code) const 
{
  switch(code) {
  case 0: return getVal() ; 
  case 1: 
    {
      Double_t c = 2./width;
      return c*(atan(c*(x.max()-mean)) - atan(c*(x.min()-mean)));
    }
  }
  
  assert(0) ;
  return 0 ;
}

