/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooNonCPEigenDecay.cc,v 1.4 2002/04/10 01:19:36 hoecker Exp $
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
 *   Apr-2002   AH allow prompt (ie, non-Pdf) mischarge treatment
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
					RooAbsReal&     wQ,
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
    _wQ       ( "wQ",       "mischarge",          this, wQ       ),
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
  
  // dummy mischarge (must be set to zero!)
  _wQ = RooRealProxy( "wQ", "mischarge", this, *(new RooRealVar( "wQ", "wQ", 0 )) );

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
    _wQ       ( "wQ",       this, other._wQ       ),
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
      return (1 + rhoQc*_acp*(1 - 2*_wQ))*(1 + 0.5*_tag*(-2*_delW));
    else
      return 1;
  }

  if (basisIndex == _basisSin) {
    
    if (rhoQc == -1) 

      return + ((1 - _acp)*_a_sin_m*(1 - _wQ) + (1 + _acp)*_a_sin_p*_wQ)*(1 - 2*_avgW)*_tag;

    else if (rhoQc == +1)

      return + ((1 + _acp)*_a_sin_p*(1 - _wQ) + (1 - _acp)*_a_sin_m*_wQ)*(1 - 2*_avgW)*_tag;

    else 
       return _tag*((_a_sin_p + _a_sin_m)/2)*(1 - 2*_avgW);
  }

  if (basisIndex == _basisCos) {
    
    if ( rhoQc == -1) 

      return - ((1 - _acp)*_a_cos_m*(1 - _wQ) + (1 + _acp)*_a_cos_p*_wQ)*(1 - 2*_avgW)*_tag;

    else if (rhoQc == +1)

      return - ((1 + _acp)*_a_cos_p*(1 - _wQ) + (1 - _acp)*_a_cos_m*_wQ)*(1 - 2*_avgW)*_tag;

    else 
      return - _tag*((_a_cos_p + _a_cos_m)/2)*(1 - 2*_avgW);
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
  Int_t rhoQc = _rhoQ*int(_correctQ);

  // correct for the right/wrong charge...
  switch(code) {

    // No integration
  case 0: return coefficient(basisIndex);

    // Integration over 'tag'
  case 1:
    if (basisIndex == _basisExp) return 2*(1 + rhoQc*_acp*(1 - 2*_wQ));
    if (basisIndex == _basisSin || basisIndex==_basisCos) return 0;
    assert( kFALSE );

    // Integration over 'rhoQ'
  case 2:
    if (basisIndex == _basisExp) return 2*(1 + 0.5*_tag*(-2.*_delW));

    if (basisIndex == _basisSin)

      return + ( (1 - _acp)*_a_sin_m + (1 + _acp)*_a_sin_p )*(1 - 2*_avgW)*_tag; 

    if (basisIndex == _basisCos)

      return - ( (1 - _acp)*_a_cos_m + (1 + _acp)*_a_cos_p )*(1 - 2*_avgW)*_tag; 

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
	   << ", tag dilution: " << (1 - 2*_avgW)
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
  // maximum probability density 
  double a1 = 1 + sqrt(pow(_a_cos_m, 2) + pow(_a_sin_m, 2));
  double a2 = 1 + sqrt(pow(_a_cos_p, 2) + pow(_a_sin_p, 2));
 
  Double_t maxAcceptProb = (1 + fabs(_acp)) * (a1 > a2 ? a1 : a2);

  // Generate delta-t dependent
  while (kTRUE) {

    // B flavor and rho charge (we do not use the integrated weights)
    _tag  = (RooRandom::uniform()<=0.5) ? 1 : -1;
    _rhoQ = (RooRandom::uniform()<=0.5) ? 1 : -1;

    // opposite charge?
    Int_t rhoQc = _rhoQ*int(_correctQ);

    Double_t rand = RooRandom::uniform();
    Double_t tval(0);

    switch(_type) {

    case SingleSided:
      tval = -_tau*log(rand);
      break;

    case Flipped:
      tval = +_tau*log(rand);
      break;

    case DoubleSided:
      tval = (rand<=0.5) ? -_tau*log(2*rand) : +_tau*log(2*(rand-0.5));
      break;
    }

    // get coefficients
    Double_t expC = coefficient( _basisExp );
    Double_t sinC = coefficient( _basisSin );
    Double_t cosC = coefficient( _basisCos );
    
    // probability density
    Double_t acceptProb  = expC + sinC*sin(_dm*tval) + cosC*cos(_dm*tval);

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

