/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

/**
\file RooTruthModel.cxx
\class RooTruthModel
\ingroup Roofitcore

RooTruthModel is an implementation of RooResolution
model that provides a delta-function resolution model.
The truth model supports <i>all</i> basis functions because it evaluates each basis function as
as a RooFormulaVar.  The 6 basis functions used in B mixing and decay and 2 basis
functions used in D mixing have been hand coded for increased execution speed.
**/

#include "RooFit.h"

#include "Riostream.h"
#include "RooTruthModel.h"
#include "RooGenContext.h"
#include "RooAbsAnaConvPdf.h"

#include "TError.h"

#include <algorithm>
using namespace std ;

ClassImp(RooTruthModel);
;



////////////////////////////////////////////////////////////////////////////////
/// Constructor of a truth resolution model, i.e. a delta function in observable 'xIn'

RooTruthModel::RooTruthModel(const char *name, const char *title, RooAbsRealLValue& xIn) :
  RooResolutionModel(name,title,xIn)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooTruthModel::RooTruthModel(const RooTruthModel& other, const char* name) :
  RooResolutionModel(other,name)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooTruthModel::~RooTruthModel()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Return basis code for given basis definition string. Return special
/// codes for 'known' bases for which compiled definition exists. Return
/// generic bases code if implementation relies on TFormula interpretation
/// of basis name

Int_t RooTruthModel::basisCode(const char* name) const
{
  // Check for optimized basis functions
  if (!TString("exp(-@0/@1)").CompareTo(name)) return expBasisPlus ;
  if (!TString("exp(@0/@1)").CompareTo(name)) return expBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)").CompareTo(name)) return expBasisSum ;
  if (!TString("exp(-@0/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisPlus ;
  if (!TString("exp(@0/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisSum ;
  if (!TString("exp(-@0/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisPlus ;
  if (!TString("exp(@0/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisSum ;
  if (!TString("(@0/@1)*exp(-@0/@1)").CompareTo(name)) return linBasisPlus ;
  if (!TString("(@0/@1)*(@0/@1)*exp(-@0/@1)").CompareTo(name)) return quadBasisPlus ;
  if (!TString("exp(-@0/@1)*cosh(@0*@2/2)").CompareTo(name)) return coshBasisPlus;
  if (!TString("exp(@0/@1)*cosh(@0*@2/2)").CompareTo(name)) return coshBasisMinus;
  if (!TString("exp(-abs(@0)/@1)*cosh(@0*@2/2)").CompareTo(name)) return coshBasisSum;
  if (!TString("exp(-@0/@1)*sinh(@0*@2/2)").CompareTo(name)) return sinhBasisPlus;
  if (!TString("exp(@0/@1)*sinh(@0*@2/2)").CompareTo(name)) return sinhBasisMinus;
  if (!TString("exp(-abs(@0)/@1)*sinh(@0*@2/2)").CompareTo(name)) return sinhBasisSum;

  // Truth model is delta function, i.e. convolution integral
  // is basis function, therefore we can handle any basis function
  return genericBasis ;
}




////////////////////////////////////////////////////////////////////////////////
/// Changes associated bases function to 'inBasis'

void RooTruthModel::changeBasis(RooFormulaVar* inBasis)
{
  // Process change basis function. Since we actually
  // evaluate the basis function object, we need to
  // adjust our client-server links to the basis function here

  // Remove client-server link to old basis
  if (_basis) {
    removeServer(*_basis) ;
  }

  // Change basis pointer and update client-server link
  _basis = inBasis ;
  if (_basis) {
    addServer(*_basis,kTRUE,kFALSE) ;
  }

  _basisCode = inBasis?basisCode(inBasis->GetTitle()):0 ;
}





////////////////////////////////////////////////////////////////////////////////
/// Evaluate the truth model: a delta function when used as PDF,
/// the basis function itself, when convoluted with a basis function.

Double_t RooTruthModel::evaluate() const
{
  // No basis: delta function
  if (_basisCode == noBasis) {
    if (x==0) return 1 ;
    return 0 ;
  }

  // Generic basis: evaluate basis function object
  if (_basisCode == genericBasis) {
    return basis().getVal() ;
  }

  // Precompiled basis functions
  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  // Enforce sign compatibility
  if ((basisSign==Minus && x>0) ||
      (basisSign==Plus  && x<0)) return 0 ;


  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
  // Return desired basis function
  switch(basisType) {
  case expBasis: {
    //cout << " RooTruthModel::eval(" << GetName() << ") expBasis mode ret = " << exp(-fabs((Double_t)x)/tau) << " tau = " << tau << endl ;
    return exp(-fabs((Double_t)x)/tau) ;
  }
  case sinBasis: {
    Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
    return exp(-fabs((Double_t)x)/tau)*sin(x*dm) ;
  }
  case cosBasis: {
    Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
    return exp(-fabs((Double_t)x)/tau)*cos(x*dm) ;
  }
  case linBasis: {
    Double_t tscaled = fabs((Double_t)x)/tau;
    return exp(-tscaled)*tscaled ;
  }
  case quadBasis: {
    Double_t tscaled = fabs((Double_t)x)/tau;
    return exp(-tscaled)*tscaled*tscaled;
  }
  case sinhBasis: {
    Double_t dg = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
    return exp(-fabs((Double_t)x)/tau)*sinh(x*dg/2) ;
  }
  case coshBasis: {
    Double_t dg = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
    return exp(-fabs((Double_t)x)/tau)*cosh(x*dg/2) ;
  }
  default:
    R__ASSERT(0) ;
  }

  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Advertise analytical integrals for compiled basis functions and when used
/// as p.d.f without basis function.

Int_t RooTruthModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  switch(_basisCode) {

  // Analytical integration capability of raw PDF
  case noBasis:
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;

  // Analytical integration capability of convoluted PDF
  case expBasisPlus:
  case expBasisMinus:
  case expBasisSum:
  case sinBasisPlus:
  case sinBasisMinus:
  case sinBasisSum:
  case cosBasisPlus:
  case cosBasisMinus:
  case cosBasisSum:
  case linBasisPlus:
  case quadBasisPlus:
  case sinhBasisPlus:
  case sinhBasisMinus:
  case sinhBasisSum:
  case coshBasisPlus:
  case coshBasisMinus:
  case coshBasisSum:
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;
  }

  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrals when used as p.d.f and for compiled
/// basis functions.

Double_t RooTruthModel::analyticalIntegral(Int_t code, const char* rangeName) const
{

  // Code must be 1
  R__ASSERT(code==1) ;

  // Unconvoluted PDF
  if (_basisCode==noBasis) return 1 ;

  // Precompiled basis functions
  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;
  //cout << " calling RooTruthModel::analyticalIntegral with basisType " << basisType << endl;

  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
  switch (basisType) {
  case expBasis:
    {
      // WVE fixed for ranges
      Double_t result(0) ;
      if (tau==0) return 1 ;
      if ((basisSign != Minus) && (x.max(rangeName)>0)) {
   result += tau*(-exp(-x.max(rangeName)/tau) -  -exp(-max(0.,x.min(rangeName))/tau) ) ; // plus and both
      }
      if ((basisSign != Plus) && (x.min(rangeName)<0)) {
   result -= tau*(-exp(-max(0.,x.min(rangeName))/tau)) - -tau*exp(-x.max(rangeName)/tau) ;   // minus and both
      }

      return result ;
    }
  case sinBasis:
    {
      Double_t result(0) ;
      if (tau==0) return 0 ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      if (basisSign != Minus) result += exp(-x.max(rangeName)/tau)*(-1/tau*sin(dm*x.max(rangeName)) - dm*cos(dm*x.max(rangeName))) + dm;  // fixed FMV 08/29/03
      if (basisSign != Plus)  result -= exp( x.min(rangeName)/tau)*(-1/tau*sin(dm*(-x.min(rangeName))) - dm*cos(dm*(-x.min(rangeName)))) + dm ;  // fixed FMV 08/29/03
      return result / (1/(tau*tau) + dm*dm) ;
    }
  case cosBasis:
    {
      Double_t result(0) ;
      if (tau==0) return 1 ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      if (basisSign != Minus) result += exp(-x.max(rangeName)/tau)*(-1/tau*cos(dm*x.max(rangeName)) + dm*sin(dm*x.max(rangeName))) + 1/tau ;
      if (basisSign != Plus)  result += exp( x.min(rangeName)/tau)*(-1/tau*cos(dm*(-x.min(rangeName))) + dm*sin(dm*(-x.min(rangeName)))) + 1/tau ; // fixed FMV 08/29/03
      return result / (1/(tau*tau) + dm*dm) ;
    }
  case linBasis:
    {
      if (tau==0) return 0 ;
      Double_t t_max = x.max(rangeName)/tau ;
      return tau*( 1 - (1 + t_max)*exp(-t_max) ) ;
    }
  case quadBasis:
    {
      if (tau==0) return 0 ;
      Double_t t_max = x.max(rangeName)/tau ;
      return tau*( 2 - (2 + (2 + t_max)*t_max)*exp(-t_max) ) ;
    }
  case sinhBasis:
    {
      Double_t result(0) ;
      if (tau==0) return 0 ;
      Double_t dg = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      Double_t taup = 2*tau/(2-tau*dg);
      Double_t taum = 2*tau/(2+tau*dg);
      if (basisSign != Minus) result += 0.5*( taup*(1-exp(-x.max(rangeName)/taup)) - taum*(1-exp(-x.max(rangeName)/taum)) ) ;
      if (basisSign != Plus)  result -= 0.5*( taup*(1-exp( x.min(rangeName)/taup)) - taum*(1-exp( x.min(rangeName)/taum)) ) ;
      return result ;
    }
  case coshBasis:
    {
      Double_t result(0) ;
      if (tau==0) return 1 ;
      Double_t dg = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      Double_t taup = 2*tau/(2-tau*dg);
      Double_t taum = 2*tau/(2+tau*dg);
      if (basisSign != Minus) result += 0.5*( taup*(1-exp(-x.max(rangeName)/taup)) + taum*(1-exp(-x.max(rangeName)/taum)) ) ;
      if (basisSign != Plus)  result += 0.5*( taup*(1-exp( x.min(rangeName)/taup)) + taum*(1-exp( x.min(rangeName)/taum)) ) ;
      return result ;
    }
  default:
    R__ASSERT(0) ;
  }

  R__ASSERT(0) ;
  return 0 ;
}


////////////////////////////////////////////////////////////////////////////////

RooAbsGenContext* RooTruthModel::modelGenContext
(const RooAbsAnaConvPdf& convPdf, const RooArgSet &vars, const RooDataSet *prototype,
 const RooArgSet* auxProto, Bool_t verbose) const
{
  RooArgSet forceDirect(convVar()) ;
  return new RooGenContext(dynamic_cast<const RooAbsPdf&>(convPdf), vars, prototype,
                           auxProto, verbose, &forceDirect) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Advertise internal generator for observable x

Int_t RooTruthModel::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implement internal generator for observable x,
/// x=0 for all events following definition
/// of delta function

void RooTruthModel::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;
  Double_t zero(0.) ;
  x = zero ;
  return;
}
