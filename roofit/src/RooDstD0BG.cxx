/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooDstD0BG.rdl,v 1.4 2001/01/23 19:36:16 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   UE, Ulrik Egede, RAL, U.Egede@rl.ac.uk
 *   MT, Max Turri, UC Santa Cruz
 *   CC, Chih-hsiang Cheng, Stanford University
 * History:
 *   07-Feb-2000 DK Created initial version from RooGaussianProb
 *   29-Feb-2000 UE Created as copy of RooArgusBG.rdl
 *   12-Jul-2000 MT Implement alpha parameter
 *   21-Aug-2001 CC Migrate from RooFitTool  
 *
 * Description : Background shape for D*-D0 mass difference
 *
 * Copyright (C) 2000 RAL
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooDstD0BG.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooDstD0BG) 

static const char rcsid[] =
"$Id: RooDstD0BG.cc,v 1.6 2001/01/23 19:36:16 verkerke Exp $";

RooDstD0BG::RooDstD0BG(const char *name, const char *title,
		       RooAbsReal& _dm, RooAbsReal& _dm0,
		       RooAbsReal& _c) :
  RooAbsPdf(name,title),
  dm("dm","Dstar-D0 Mass Diff",this, _dm),
  dm0("dm0","Threshold",this, _dm0),
  c("c","Shape Parameter",this, _c)
{
}

RooDstD0BG::RooDstD0BG(const RooDstD0BG& other, const char *name) :
  RooAbsPdf(other,name), dm("dm",this,other.dm), dm0("dm0",this,other.dm0),
  c("c",this,other.c)
{
}

Double_t RooDstD0BG::evaluate(const RooArgSet *nset) const
{
  Double_t arg= dm- dm0;
  if (arg <= 0 ) return 0;

  return 1- exp(-arg/c);
}

Int_t RooDstD0BG::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,dm)) return 1 ;
  return 0 ;
}

Double_t RooDstD0BG::analyticalIntegral(Int_t code) const 
{
  switch(code) {
  case 0: return getVal() ; 
  case 1: 
    {
      Double_t min= dm.min();
      Double_t max= dm.max();
      if ( min > dm0 ) 
	return c*(exp(-(max-dm0)/c)-exp(-(min-dm0)/c)) + max-min;
      else if ( max > dm0 )
	return c*(exp(-(max-dm0)/c)-1) + max- dm0;
      else
	return 0;
    }
  }
  
  assert(0) ;
  return 0 ;
}
