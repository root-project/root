/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooNonCPEigenDecay.rdl,v 1.3 2002/04/09 00:15:09 hoecker Exp $
 * Authors:
 *   AH, Andreas Hoecker, Orsay, hoecker@slac.stanford.edu
 *   SL, Sandrine Laplace, Orsay, laplace@slac.stanford.edu
 *   JS, Jan Stark, Paris, stark@slac.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   Nov-2001   WV Created initial version
 *   Mar-2002   JS Commit improved version to CVS
 *   May-2002   JS Changed the set of CP parameters (mathematically equivalent)
 *
 * Copyright (C) 2002 University of California, IN2P3
 *****************************************************************************/
#ifndef ROO_NONCPEIGEN_DECAY
#define ROO_NONCPEIGEN_DECAY

#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooCategoryProxy.hh"

class RooNonCPEigenDecay : public RooConvolutedPdf {

public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooNonCPEigenDecay( void ) { }

  // with explicit mischarge
  RooNonCPEigenDecay( const char *name, const char *title, 
		      RooRealVar&     t, 
	              RooAbsCategory& tag,
		      RooAbsReal&     tau, 
		      RooAbsReal&     dm,
		      RooAbsReal&     avgW, 
		      RooAbsReal&     delW, 
		      RooAbsCategory& rhoQ, 
		      RooAbsReal&     correctQ, 
		      RooAbsReal&     wQ,
		      RooAbsReal&     a,
		      RooAbsReal&     C,
		      RooAbsReal&     delC,
		      RooAbsReal&     S,
		      RooAbsReal&     delS,
		      const RooResolutionModel& model, 
		      DecayType       type = DoubleSided );

  // no explicit mischarge (=> set to zero)
  RooNonCPEigenDecay( const char *name, const char *title, 
		      RooRealVar&     t, 
	              RooAbsCategory& tag,
		      RooAbsReal&     tau, 
		      RooAbsReal&     dm,
		      RooAbsReal&     avgW, 
		      RooAbsReal&     delW, 
		      RooAbsCategory& rhoQ, 
		      RooAbsReal&     correctQ, 
		      RooAbsReal&     a,
		      RooAbsReal&     C,
		      RooAbsReal&     delC,
		      RooAbsReal&     S,
		      RooAbsReal&     delS,
		      const RooResolutionModel& model, 
		      DecayType       type = DoubleSided );

  RooNonCPEigenDecay(const RooNonCPEigenDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { 
    return new RooNonCPEigenDecay(*this,newname); 
  }
  virtual ~RooNonCPEigenDecay( void );

  virtual Double_t coefficient( Int_t basisIndex ) const;

  virtual Int_t getCoefAnalyticalIntegral( RooArgSet& allVars, 
  					   RooArgSet& analVars ) const;
  virtual Double_t coefAnalyticalIntegral( Int_t coef, Int_t code ) const;

  Int_t getGenerator( const RooArgSet& directVars, 
		      RooArgSet&       generateVars ) const;
  void initGenerator( Int_t code );
  void generateEvent( Int_t code );

protected:

  RooRealProxy     _acp;
  RooRealProxy     _C;
  RooRealProxy     _delC;
  RooRealProxy     _S;
  RooRealProxy     _delS;
  RooRealProxy     _avgW;
  RooRealProxy     _delW;
  RooRealProxy     _t;
  RooRealProxy     _tau;
  RooRealProxy     _dm;
  RooCategoryProxy _tag;
  RooCategoryProxy _rhoQ;
  RooRealProxy     _correctQ;
  RooRealProxy     _wQ;
  Double_t         _genB0Frac;
  Double_t         _genRhoPlusFrac;
  
  DecayType        _type;
  Int_t            _basisExp;
  Int_t            _basisSin;
  Int_t            _basisCos;

  ClassDef(RooNonCPEigenDecay,1) // PDF to model CP-violating decays to final states 
                                 // which are not CP eigenstates
};

#endif
