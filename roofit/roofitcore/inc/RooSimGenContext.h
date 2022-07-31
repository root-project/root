/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSimGenContext.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SIM_GEN_CONTEXT
#define ROO_SIM_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"
#include <vector>

class RooSimultaneous;
class RooDataSet;
class RooAbsCategoryLValue ;

class RooSimGenContext : public RooAbsGenContext {
public:
  RooSimGenContext(const RooSimultaneous &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
                   const RooArgSet* auxProto=0, bool _verbose= false);
  ~RooSimGenContext() override;
  void setProtoDataOrder(Int_t* lut) override ;

  void attach(const RooArgSet& params) override ;

  void printMultiline(std::ostream &os, Int_t content, bool verbose=false, TString indent="") const override ;


protected:

  void initGenerator(const RooArgSet &theEvent) override;
  void generateEvent(RooArgSet &theEvent, Int_t remaining) override;

  RooDataSet* createDataSet(const char* name, const char* title, const RooArgSet& obs) override ;
  void updateFractions() ;

  RooSimGenContext(const RooSimGenContext& other) ;

  RooAbsCategoryLValue* _idxCat{nullptr};    ///< Clone of index category
  RooArgSet*            _idxCatSet{nullptr}; ///< Owner of index category components
  const RooDataSet *_prototype{nullptr};     ///< Prototype data set
  const RooSimultaneous *_pdf{nullptr};      ///< Original PDF
  std::vector<RooAbsGenContext*> _gcList ;   ///< List of component generator contexts
  std::vector<int>               _gcIndex ;  ///< Index value corresponding to component
  bool _haveIdxProto{false};               ///< Flag set if generation of index is requested
  TString _idxCatName{};                     ///< Name of index category
  Int_t _numPdf{0};                          ///< Number of generated PDFs
  double* _fracThresh{nullptr};            ///<[_numPdf] Fraction threshold array
  RooDataSet* _protoData{nullptr};           ///<! Prototype dataset

  RooArgSet _allVarsPdf{};        ///< All pdf variables

  ClassDefOverride(RooSimGenContext,0) // Context for efficiently generating a dataset from a RooSimultaneous PDF
};

#endif
