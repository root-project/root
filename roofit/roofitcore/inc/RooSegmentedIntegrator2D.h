/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSegmentedIntegrator2D.h,v 1.7 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SEGMENTED_INTEGRATOR_2D
#define ROO_SEGMENTED_INTEGRATOR_2D

#include "RooSegmentedIntegrator1D.h"
#include "RooIntegrator1D.h"
#include "RooNumIntConfig.h"

class RooSegmentedIntegrator2D : public RooSegmentedIntegrator1D {
public:

  // Constructors, assignment etc
  RooSegmentedIntegrator2D() ;

  RooSegmentedIntegrator2D(const RooAbsFunc& function, const RooNumIntConfig& config) ;
  RooSegmentedIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax,
        const RooNumIntConfig& config) ;
  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;
  ~RooSegmentedIntegrator2D() override ;

  bool checkLimits() const override;

  bool canIntegrate1D() const override { return false ; }
  bool canIntegrate2D() const override { return true ; }
  bool canIntegrateND() const override { return false ; }
  bool canIntegrateOpenEnded() const override { return false ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;

  RooSegmentedIntegrator1D* _xIntegrator ;
  RooAbsFunc* _xint ;

  ClassDefOverride(RooSegmentedIntegrator2D,0) // 2-dimensional piece-wise numerical integration engine
};

#endif
