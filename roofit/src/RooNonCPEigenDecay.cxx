/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooNonCPEigenDecay.cc,v 1.1 2002/03/10 21:36:32 stark Exp $
 * Authors:
 *   AH, Andreas Hoecker, Orsay, hoecker@slac.stanford.edu
 *   SL, Sandrine Laplace, Orsay, laplace@slac.stanford.edu
 *   JS, Jan Stark, Paris, stark@slac.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   Nov-2001   WV Created initial version
 *   Dec-2001   SL mischarge correction, direct CPV
 *   Jan-2002   AH built dedicated generator + code cleaning
 *   Mar-2002   JS committed debugged version to CVS
 *
 * Copyright (C) 2002 University of California, IN2P3
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// Time-dependent RooConvolutedPdf for CP violating decays 
// to Non-CP eigenstates (eg, B0 -> rho+-pi-+).
// For a description of the physics model as well as the
// definition of the parameters see the BaBar Physics Book,
// section 6.5.2.3
// 

#include <iostream.h>
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRandom.hh"
#include "RooFitModels/RooNonCPEigenDecay.hh"

ClassImp(RooNonCPEigenDecay);

#define Debug_RooNonCPEigenDecay 1

RooNonCPEigenDecay::RooNonCPEigenDecay( const char *name, const char *title, 
					RooRealVar&     t, 
					RooAbsCategory& tag,
					RooAbsReal&     tau, 
					RooAbsReal&     dm,
					RooAbsReal&     avgW, 
					RooAbsReal&     delW, 
					RooAbsCategory& rhoQ, 
					RooAbsReal&     correctQ,
					RooAbsReal&     acp,
					RooAbsReal&     a_cos_p,
					RooAbsReal&     a_cos_m,
					RooAbsReal&     a_sin_p,
					RooAbsReal&     a_sin_m,
					const RooResolutionModel& model, 
					DecayType       type )
  : RooConvolutedPdf( name, title, model, t ), 
  _rhoQ     ( "rhoQ",     "Charge of the rho",  this, rhoQ     ),
  _correctQ ( "correctQ", "correction of rhoQ", this, correctQ ),
  _acp      ( "acp",      "acp",                this, acp      ),
  _a_cos_p  ( "a_cos_p",  "a+(cos)",            this, a_cos_p  ),
  _a_cos_m  ( "a_cos_m",  "a-(cos)",            this, a_cos_m  ),
  _a_sin_p  ( "a_sin_p",  "a+(sin)",            this, a_sin_p  ),
  _a_sin_m  ( "a_sin_m",  "a-(sin)",            this, a_sin_m  ),
  _avgW     ( "avgW",     "Average mistag rate",this, avgW     ),
  _delW     ( "delW",     "Shift mistag rate",  this, delW     ),
  _tag      ( "tag",      "CP state",           this, tag      ),
  _tau      ( "tau",      "decay time",         this, tau      ),
  _dm       ( "dm",       "mixing frequency",   this, dm       ),
  _t        ( "t",        "time",               this, t        ),
  _type     ( type ),
  _genB0Frac     ( 0 ),
  _genRhoPlusFrac( 0 )
{

  // Constructor
  switch(type) {
  case SingleSided:
    _basisExp = declareBasis( "exp(-@0/@1)",            RooArgList( tau     ) );
    _basisSin = declareBasis( "exp(-@0/@1)*sin(@0*@2)", RooArgList( tau, dm ) );
    _basisCos = declareBasis( "exp(-@0/@1)*cos(@0*@2)", RooArgList( tau, dm ) );
    break;
  case Flipped:
    _basisExp = declareBasis( "exp(@0)/@1)",            RooArgList( tau     ) );
    _basisSin = declareBasis( "exp(@0/@1)*sin(@0*@2)",  RooArgList( tau, dm ) );
    _basisCos = declareBasis( "exp(@0/@1)*cos(@0*@2)",  RooArgList( tau, dm ) );
    break;
  case DoubleSided:
    _basisExp = declareBasis( "exp(-abs(@0)/@1)",            RooArgList( tau     ) );
    _basisSin = declareBasis( "exp(-abs(@0)/@1)*sin(@0*@2)", RooArgList( tau, dm ) );
    _basisCos = declareBasis( "exp(-abs(@0)/@1)*cos(@0*@2)", RooArgList( tau, dm ) );
    break;
  }
}

RooNonCPEigenDecay::RooNonCPEigenDecay( const RooNonCPEigenDecay& other, const char* name ) 
  : RooConvolutedPdf( other, name ), 
  _rhoQ     ( "rhoQ",     this, other._rhoQ     ),
  _correctQ ( "correctQ", this, other._correctQ ),
  _acp      ( "acp",      this, other._acp      ),
  _a_cos_p  ( "a_cos_p",  this, other._a_cos_p  ),
  _a_cos_m  ( "a_cos_m",  this, other._a_cos_m  ),
  _a_sin_p  ( "a_sin_p",  this, other._a_sin_p  ),
  _a_sin_m  ( "a_sin_m",  this, other._a_sin_m  ),
  _avgW     ( "avgW",     this, other._avgW     ),
  _delW     ( "delW",     this, other._delW     ),
  _tag      ( "tag",      this, other._tag      ),
  _tau      ( "tau",      this, other._tau      ),
  _dm       ( "dm",       this, other._dm       ),
  _t        ( "t",        this, other._t        ),
  _type          ( other._type           ),
  _basisExp      ( other._basisExp       ),
  _basisSin      ( other._basisSin       ),
  _basisCos      ( other._basisCos       ),
  _genB0Frac     ( other._genB0Frac      ),
  _genRhoPlusFrac( other._genRhoPlusFrac )
{
  // Copy constructor
}

RooNonCPEigenDecay::~RooNonCPEigenDecay( void )
{
  // Destructor
}

Double_t RooNonCPEigenDecay::coefficient( Int_t basisIndex ) const 
{
  // B0    : _tag  == +1 
  // B0bar : _tag  == -1 
  // rho+  : _rhoQ == +1
  // rho-  : _rhoQ == -1
  // the charge corrrection factor "_correctQ" serves to implement mis-charges

  Int_t rhoQc = _rhoQ * int(_correctQ);
  assert( rhoQc == 1 || rhoQc == -1 );

  if (basisIndex == _basisExp) {
    if (rhoQc == -1 || rhoQc == +1) 
      return (1.0 + rhoQc*_acp)*(1 + 0.5*_tag*(-2.*_delW));
    else
      return 1.0;
  }

  if (basisIndex == _basisSin) {
    
    if (rhoQc == -1) 
      return + (1.0 - _acp) * (_a_sin_m * (1.-2.*_avgW))*_tag;
    else if (rhoQc == +1)
      return + (1.0 + _acp) * (_a_sin_p * (1.-2.*_avgW))*_tag;
    else 
       return _tag * ((_a_sin_p + _a_sin_m)/2) * (1.-2.*_avgW);
  }

  if (basisIndex == _basisCos) {
    
    if ( rhoQc == -1) 
      return - (1.0 - _acp) * (_a_cos_m * (1.-2.*_avgW))*_tag;
    else if (rhoQc == +1)
      return - (1.0 + _acp) * (_a_cos_p * (1.-2.*_avgW))*_tag;
    else 
      return -_tag * ((_a_cos_p + _a_cos_m)/2) * (1.-2.*_avgW);
  }

  return 0;
}

// advertise analytical integration
Int_t RooNonCPEigenDecay::getCoefAnalyticalIntegral( RooArgSet& allVars, 
						     RooArgSet& analVars ) const 
{
  if (matchArgs( allVars, analVars, _tag, _rhoQ )) return 3;
  if (matchArgs( allVars, analVars, _rhoQ       )) return 2;
  if (matchArgs( allVars, analVars, _tag        )) return 1;

  return 0;
}

Double_t RooNonCPEigenDecay::coefAnalyticalIntegral( Int_t basisIndex, 
						     Int_t code ) const 
{
  Int_t rhoQc = _rhoQ * int(_correctQ);

  // correct for the right/wrong charge...
  switch(code) {

    // No integration
  case 0: return coefficient(basisIndex);

    // Integration over 'tag'
  case 1:
    if (basisIndex == _basisExp) return 2*(1.0 + rhoQc*_acp);
    if (basisIndex == _basisSin || basisIndex==_basisCos) return 0;
    assert( kFALSE );

    // Integration over 'rhoQ'
  case 2:
    if (basisIndex == _basisExp) return 2*(1 + 0.5*_tag*(-2.*_delW));
    if (basisIndex == _basisSin)
      return + ( (1.0 - _acp)*_a_sin_m + (1.0 + _acp)*_a_sin_p )*(1.-2.*_avgW)*_tag; 
    if (basisIndex == _basisCos)
      return - ( (1.0 - _acp)*_a_cos_m + (1.0 + _acp)*_a_cos_p )*(1.-2.*_avgW)*_tag; 
    assert( kFALSE );

    // Integration over 'tag' and 'rhoQ'
  case 3:
    if (basisIndex == _basisExp) return 2*2; // for both: tag and charge
    if (basisIndex == _basisSin || basisIndex==_basisCos) return 0;
    assert( kFALSE );

  default:
    assert( kFALSE );
  }
    
  return 0;
}

Int_t RooNonCPEigenDecay::getGenerator( const RooArgSet& directVars, 
					RooArgSet&       generateVars ) const
{
  if (matchArgs( directVars, generateVars, _t, _tag, _rhoQ )) return 4;  
  if (matchArgs( directVars, generateVars, _t, _rhoQ       )) return 3;  
  if (matchArgs( directVars, generateVars, _t, _tag        )) return 2;  
  if (matchArgs( directVars, generateVars, _t              )) return 1;  
  return 0;
}

void RooNonCPEigenDecay::initGenerator( Int_t code )
{

  if (code == 2 || code == 4) {
    // Calculate the fraction of mixed events to generate
    Double_t sumInt1 = RooRealIntegral( "sumInt1", "sum integral1", *this, 
					RooArgSet( _t.arg(), _tag.arg(), _rhoQ.arg() )
				      ).getVal();
    _tag = 1;
    Double_t b0Int1 = RooRealIntegral( "mixInt1", "mix integral1", *this,
				       RooArgSet( _t.arg(), _rhoQ.arg() )
				     ).getVal();
    _genB0Frac = b0Int1/sumInt1;

    if (Debug_RooNonCPEigenDecay == 1) 
      cout << "     o RooNonCPEigenDecay::initgenerator: genB0Frac     : " 
	   << _genB0Frac 
	   << ", tag dilution: " << (1.-2.*_avgW)
	   << endl;
  }  

  if (code == 3 || code == 4) {
    // Calculate the fraction of positive rho's to generate
    Double_t sumInt2 = RooRealIntegral( "sumInt2", "sum integral2", *this, 
					RooArgSet( _t.arg(), _tag.arg(), _rhoQ.arg() )
				      ).getVal();
    _rhoQ = 1;
    Double_t b0Int2 = RooRealIntegral( "mixInt2", "mix integral2", *this,
				       RooArgSet( _t.arg(), _tag.arg() )
				     ).getVal();
    _genRhoPlusFrac = b0Int2/sumInt2;

    if (Debug_RooNonCPEigenDecay == 1) 
      cout << "     o RooNonCPEigenDecay::initgenerator: genRhoPlusFrac: " 
	   << _genRhoPlusFrac << endl;
  }  
}


void RooNonCPEigenDecay::generateEvent( Int_t code )
{
  // Generate mix-state dependent
  if (code == 2 || code == 4) {
    Double_t rand1 = RooRandom::uniform();
    _tag = (rand1<=_genB0Frac) ? 1 : -1;
  }
  
  // Generate delta-t dependent
  while (kTRUE) {

    // Generate charge dependent
    if (code == 3 || code == 4) {
      Double_t rand2 = RooRandom::uniform();
      _rhoQ = (rand2<=_genRhoPlusFrac) ? 1 : -1;
    }

    Int_t rhoQc = _rhoQ * int(_correctQ);

    Double_t rand = RooRandom::uniform();
    Double_t tval(0);

    switch(_type) {

    case SingleSided:
      tval = -_tau*log(rand);
      break;

    case Flipped:
      tval= +_tau*log(rand);
      break;

    case DoubleSided:
      tval = (rand<=0.5) ? -_tau*log(2*rand) : +_tau*log(2*(rand-0.5));
      break;
    }

    // accept event if T is in generated range
    Double_t basisC  = (1.0 + rhoQc*_acp)*(1 + 0.5*(-2.*_delW));
    Double_t sineC   = (rhoQc == -1) ? + (1.0 - _acp) * (_a_sin_m*(1.-2.*_avgW))*_tag :
                                       + (1.0 + _acp) * (_a_sin_p*(1.-2.*_avgW))*_tag;
    Double_t cosineC = (rhoQc == -1) ? - (1.0 - _acp) * (_a_cos_m*(1.-2.*_avgW))*_tag :
                                       - (1.0 + _acp) * (_a_cos_p*(1.-2.*_avgW))*_tag;

    // maximum probability density
    Double_t maxAcceptProb = basisC + fabs(sineC) + fabs(cosineC);
    
    // current probability density
    Double_t acceptProb    = basisC + sineC*sin(_dm*tval) + cosineC*cos(_dm*tval);

    // sanity check...
    assert( acceptProb <= maxAcceptProb );

    // hit or miss...
    Bool_t accept = maxAcceptProb*RooRandom::uniform() < acceptProb ? kTRUE : kFALSE;
    
    if (accept && tval<_t.max() && tval>_t.min()) {
      _t = tval;
      break;
    }
  }  
}

