/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGlobalFunc.rdl,v 1.2 2005/02/15 21:16:44 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_GLOBAL_FUNC
#define ROO_GLOBAL_FUNC

#include "RooFitCore/RooCmdArg.hh"
#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooAbsPdf.hh"
class RooAbsData ;
class RooArgSet ;
class RooCategory ;
class RooAbsReal ;
class RooAbsBinning ;
class RooAbsPdf ;
class RooConstVar ;

// RooAbsReal::plotOn arguments
RooCmdArg DrawOption(const char* opt) ;
RooCmdArg Normalization(Double_t scaleFactor) ;
RooCmdArg Slice(const RooArgSet& sliceSet) ;
RooCmdArg Project(const RooArgSet& projSet) ;
RooCmdArg ProjWData(const RooAbsData& projData) ;
RooCmdArg ProjWData(const RooArgSet& projSet, const RooAbsData& projData) ;
RooCmdArg Asymmetry(const RooCategory& cat) ;
RooCmdArg Precision(Double_t prec) ;
RooCmdArg ShiftToZero() ;
RooCmdArg Range(const char* rangeName) ;
RooCmdArg Range(Double_t lo, Double_t hi) ;
RooCmdArg VLines() ;
RooCmdArg LineColor(Color_t color) ;
RooCmdArg LineStyle(Style_t style) ;
RooCmdArg LineWidth(Width_t width) ;
RooCmdArg FillColor(Color_t color) ;
RooCmdArg FillStyle(Style_t style) ;
RooCmdArg ProjectionRange(const char* rangeName) ;
RooCmdArg Name(const char* name) ;
RooCmdArg Invisible() ;
RooCmdArg AddTo(const char* name, double wgtSel=1.0, double wgtOther=1.0) ;

// RooAbsPdf::plotOn arguments
RooCmdArg Normalization(Double_t scaleFactor, RooAbsPdf::ScaleType scaleType) ;
RooCmdArg Components(const RooArgSet& compSet) ;
RooCmdArg Components(const char* compSpec) ;

// RooAbsData::plotOn arguments
RooCmdArg Cut(const char* cutSpec) ;
RooCmdArg Cut(const RooFormulaVar& cutVar) ;
RooCmdArg Binning(const RooAbsBinning& binning) ;
RooCmdArg MarkerStyle(Style_t style) ;
RooCmdArg MarkerSize(Size_t size) ;
RooCmdArg MarkerColor(Color_t color) ;
RooCmdArg CutRange(const char* rangeName) ;
RooCmdArg XErrorSize(Double_t width) ;
RooCmdArg RefreshNorm() ;

// RooChi2Var::ctor arguments
RooCmdArg Extended() ;
RooCmdArg DataError(RooDataHist::ErrorType) ;
RooCmdArg NumCPU(Int_t nCPU) ;

// RooAbsPdf::printLatex arguments
RooCmdArg Columns(Int_t ncol) ;
RooCmdArg OutputFile(const char* fileName) ;
RooCmdArg Format(const char* format, Int_t sigDigit=1) ;
RooCmdArg Sibling(const RooAbsCollection& sibling) ;

// RooAbsRealLValue::frame arguments
RooCmdArg Title(const char* name) ;
RooCmdArg Bins(Int_t nbin) ;

// RooAbsData::reduce arguments
RooCmdArg SelectVars(const RooArgSet& vars) ;
RooCmdArg EventRange(Int_t nStart, Int_t nStop) ;

// RooAbsPdf::fitTo arguments
RooCmdArg FitOptions(const char* opts) ;
RooCmdArg Optimize(Bool_t flag=kTRUE) ;
RooCmdArg ProjectedObservables(const RooArgSet& set) ;
RooCmdArg Verbose(Bool_t flag=kTRUE) ;
RooCmdArg Save(Bool_t flag=kTRUE) ;
RooCmdArg Timer(Bool_t flag=kTRUE) ;
RooCmdArg Blind(Bool_t flag=kTRUE) ;
RooCmdArg Strategy(Int_t code) ;
RooCmdArg InitialHesse(Bool_t flag=kTRUE) ;
RooCmdArg Hesse(Bool_t flag=kTRUE) ;
RooCmdArg Minos(Bool_t flag=kTRUE) ;


namespace RooFitShortHand {

RooArgSet S(const RooAbsArg& v1) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6, const RooAbsArg& v7) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8) ;
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8, const RooAbsArg& v9) ;

RooArgList L(const RooAbsArg& v1) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6, const RooAbsArg& v7) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8) ;
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8, const RooAbsArg& v9) ;

RooConstVar& C(Double_t value) ;

}

#endif
