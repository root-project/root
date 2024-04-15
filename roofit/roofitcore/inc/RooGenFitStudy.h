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
#ifndef ROO_GEN_FIT_STUDY
#define ROO_GEN_FIT_STUDY

#include "RooAbsStudy.h"

class RooAbsPdf;
class RooDataSet ;
class RooAbsData ;
class RooFitResult ;
class RooPlot ;
class RooRealVar ;
class RooWorkspace ;
class RooAbsGenContext ;

#include "RooArgSet.h"
#include "RooLinkedList.h"
#include "RooAbsPdf.h"
#include <string>

class RooGenFitStudy : public RooAbsStudy {
public:

  RooGenFitStudy(const char* name=nullptr, const char* title=nullptr) ;
  RooGenFitStudy(const RooGenFitStudy& other) ;
  RooAbsStudy* clone(const char* newname="") const override { return new RooGenFitStudy(newname?newname:GetName(),GetTitle()) ; }

  void setGenConfig(const char* pdfName, const char* obsName, const RooCmdArg& arg1={},const RooCmdArg& arg2={},const RooCmdArg& arg3={}) ;
  void setFitConfig(const char* pdfName, const char* obsName, const RooCmdArg& arg1={},const RooCmdArg& arg2={},const RooCmdArg& arg3={}) ;

  bool attach(RooWorkspace& w) override ;
  bool initialize() override ;
  bool execute() override ;
  bool finalize() override ;

  void Print(Option_t *options= nullptr) const override;

 protected:


  std::string _genPdfName ;
  std::string _genObsName ;
  std::string _fitPdfName ;
  std::string _fitObsName ;
  RooLinkedList _genOpts ;
  RooLinkedList _fitOpts ;

  RooAbsPdf* _genPdf = nullptr; ///<!
  RooArgSet _genObs ; ///<!
  RooAbsPdf* _fitPdf = nullptr; ///<!
  RooArgSet _fitObs ; ///<!

  RooAbsPdf::GenSpec* _genSpec = nullptr; ///<!
  RooRealVar* _nllVar = nullptr; ///<!
  RooRealVar* _ngenVar = nullptr; ///<!
  std::unique_ptr<RooArgSet> _params; ///<!
  RooArgSet* _initParams= nullptr; ///<!

  ClassDefOverride(RooGenFitStudy,2) // Generate-and-Fit study module
} ;


#endif

