/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.rdl,v 1.6 2001/08/02 23:54:24 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   08-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_MC_INTEGRATOR
#define ROO_MC_INTEGRATOR

#include "RooFitCore/RooAbsIntegrator.hh"
#include "RooFitCore/RooGrid.hh"

class RooMCIntegrator : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  enum SamplingMode { Importance, ImportanceOnly, Stratified };
  RooMCIntegrator(const RooAbsFunc& function, SamplingMode mode= Importance, Bool_t verbose= kFALSE);
  virtual ~RooMCIntegrator();

  virtual Double_t integral();

  enum Stage { AllStages, ReuseGrid, RefineGrid };
  Double_t vegas(Stage stage, UInt_t calls, UInt_t iterations, Double_t *absError= 0);

  Double_t getAlpha() const { return _alpha; }
  void setAlpha(Double_t alpha) { _alpha= alpha; }

protected:

  RooGrid _grid;

  // control variables
  Bool_t _verbose;
  Double_t _alpha;
  Int_t _mode;

  // scratch variables preserved between calls to vegas1/2/2
  Double_t _jac,_wtd_int_sum,_sum_wgts,_chi_sum,_chisq,_result,_sigma;
  UInt_t _it_start,_it_num,_samples,_calls_per_box;

  ClassDef(RooMCIntegrator,0) // multi-dimensional numerical integration engine
};

#endif
