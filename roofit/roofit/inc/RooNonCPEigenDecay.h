/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooNonCPEigenDecay.h,v 1.13 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   AH, Andreas Hoecker,  Orsay,            hoecker@slac.stanford.edu       *
 *   SL, Sandrine Laplace, Orsay,            laplace@slac.stanford.edu       *
 *   JS, Jan Stark,        Paris,            stark@slac.stanford.edu         *
 *   WV, Wouter Verkerke,  UC Santa Barbara, verkerke@slac.stanford.edu      *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California,         *
 *                          IN2P3. All rights reserved.                      *
 *                                                                           *
 * History                                                                   *
 *   Nov-2001   WV Created initial version                                   *
 *   Dec-2001   SL mischarge correction, direct CPV                          *
 *   Jan-2002   AH built dedicated generator + code cleaning                 *
 *   Mar-2002   JS committed debugged version to CVS                         *
 *   Apr-2002   AH allow prompt (ie, non-Pdf) mischarge treatment            *
 *   May-2002   JS Changed the set of CP parameters (mathematically equiv.)  *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NONCPEIGEN_DECAY
#define ROO_NONCPEIGEN_DECAY

#include "RooAbsAnaConvPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"

class RooNonCPEigenDecay : public RooAbsAnaConvPdf {

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

  RooNonCPEigenDecay(const RooNonCPEigenDecay& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override {
    return new RooNonCPEigenDecay(*this,newname);
  }
  ~RooNonCPEigenDecay( void ) override;

  double coefficient( Int_t basisIndex ) const override;

  Int_t getCoefAnalyticalIntegral( Int_t coef, RooArgSet& allVars,
                    RooArgSet& analVars, const char* rangeName=nullptr ) const override;
  double coefAnalyticalIntegral( Int_t coef, Int_t code, const char* rangeName=nullptr ) const override;

  Int_t getGenerator( const RooArgSet& directVars,
            RooArgSet&       generateVars, bool staticInitOK=true ) const override;
  void initGenerator( Int_t code ) override;
  void generateEvent( Int_t code ) override;

protected:

  RooRealProxy     _acp ;
  RooRealProxy     _avgC ;
  RooRealProxy     _delC ;
  RooRealProxy     _avgS ;
  RooRealProxy     _delS ;
  RooRealProxy     _avgW ;
  RooRealProxy     _delW ;
  RooRealProxy     _t ;
  RooRealProxy     _tau;
  RooRealProxy     _dm;
  RooCategoryProxy _tag;
  RooCategoryProxy _rhoQ;
  RooRealProxy     _correctQ;
  RooRealProxy     _wQ; ///< dummy mischarge (must be set to zero!)
  double         _genB0Frac;
  double         _genRhoPlusFrac;

  DecayType        _type;
  Int_t            _basisExp;
  Int_t            _basisSin;
  Int_t            _basisCos;

  ClassDefOverride(RooNonCPEigenDecay,1) // PDF to model CP-violating decays to final states which are not CP eigenstates
};

#endif
