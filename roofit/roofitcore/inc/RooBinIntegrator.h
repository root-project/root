/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_BIN_INTEGRATOR
#define ROO_BIN_INTEGRATOR

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"

class RooBinIntegrator : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  RooBinIntegrator() ;

  RooBinIntegrator(const RooAbsFunc& function) ; 
  RooBinIntegrator(const RooAbsFunc& function, const RooNumIntConfig& config) ; 

  virtual RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const ;
  virtual ~RooBinIntegrator();

  virtual Bool_t checkLimits() const;
  virtual Double_t integral(const Double_t *yvec=0) ;

  using RooAbsIntegrator::setLimits ;
  Bool_t setLimits(Double_t* xmin, Double_t* xmax);
  virtual Bool_t setUseIntegrandLimits(Bool_t flag) {_useIntegrandLimits = flag ; return kTRUE ; }

  virtual Bool_t canIntegrate1D() const { return kTRUE ; }
  virtual Bool_t canIntegrate2D() const { return kFALSE ; }
  virtual Bool_t canIntegrateND() const { return kFALSE ; }
  virtual Bool_t canIntegrateOpenEnded() const { return kFALSE ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;	


  // Numerical integrator workspace
  mutable Double_t _xmin;              //! Lower integration bound
  mutable Double_t _xmax;              //! Upper integration bound
  mutable Double_t _range;             //! Size of integration range
  mutable Double_t _binWidth;             //! Size of integration range
  mutable Int_t _numBins;             //! Size of integration range

  Bool_t _useIntegrandLimits;  // If true limits of function binding are ued

  Double_t* xvec(Double_t& xx) { _x[0] = xx ; return _x ; }

  Double_t *_x ; //! do not persist

  ClassDef(RooBinIntegrator,0) // 1-dimensional numerical integration engine
};

#endif
