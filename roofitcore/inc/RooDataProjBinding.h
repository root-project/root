/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_DATA_PROJ_BINDING
#define ROO_DATA_PROJ_BINDING

#include "RooFitCore/RooRealBinding.hh"
class RooAbsReal ;
class RooAbsData ;
class RooSuperCategory ;
class Roo1DTable ;

class RooDataProjBinding : public RooRealBinding {
public:
  RooDataProjBinding(const RooAbsReal &real, const RooAbsData& data, const RooArgSet &vars, const RooArgSet* normSet=0) ;
  virtual ~RooDataProjBinding() ;

  virtual Double_t operator()(const Double_t xvector[]) const;

protected:

  mutable Bool_t _first   ;  // Bit indicating if operator() has been called yet
  const RooAbsReal* _real ;  // Real function to be projected
  const RooAbsData* _data ;  // Dataset used for projection
  const RooArgSet*  _nset ;  // Normalization set for real function

  RooSuperCategory* _superCat ;  // Supercategory constructed from _data's category variables
  Roo1DTable* _catTable ;        // Supercategory table generated from _data

  ClassDef(RooDataProjBinding,0) // RealFunc/Dataset binding for data projection of a real function
};

#endif

