/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGlobalFunc.rdl,v 1.10 2005/06/20 15:44:53 wverkerke Exp $
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
#ifndef ROO_GLOBAL_FUNC
#define ROO_GLOBAL_FUNC

#include "RooFitCore/RooCmdArg.hh"
#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealConstant.hh"

class RooAbsData ;
class RooArgSet ;
class RooCategory ;
class RooAbsReal ;
class RooAbsBinning ;
class RooAbsPdf ;
class RooConstVar ;

namespace RooFit {

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
RooCmdArg Range(const char* rangeName, Bool_t adjustNorm=kFALSE) ;
RooCmdArg Range(Double_t lo, Double_t hi, Bool_t adjustNorm=kFALSE) ;
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
RooCmdArg Binning(const char* binningName) ;
RooCmdArg Binning(Int_t nBins, Double_t xlo=0., Double_t xhi=0.) ;
RooCmdArg MarkerStyle(Style_t style) ;
RooCmdArg MarkerSize(Size_t size) ;
RooCmdArg MarkerColor(Color_t color) ;
RooCmdArg CutRange(const char* rangeName) ;
RooCmdArg XErrorSize(Double_t width) ;
RooCmdArg RefreshNorm() ;

// RooChi2Var::ctor arguments
RooCmdArg Extended(Bool_t flag=kTRUE) ;
RooCmdArg DataError(RooDataHist::ErrorType) ;
RooCmdArg NumCPU(Int_t nCPU) ;

// RooAbsPdf::printLatex arguments
RooCmdArg Columns(Int_t ncol) ;
RooCmdArg OutputFile(const char* fileName) ;
RooCmdArg Format(const char* format, Int_t sigDigit) ;
RooCmdArg Format(const char* what, const RooCmdArg& arg1=RooCmdArg::none, const RooCmdArg& arg2=RooCmdArg::none,
                 const RooCmdArg& arg3=RooCmdArg::none,const RooCmdArg& arg4=RooCmdArg::none,
                 const RooCmdArg& arg5=RooCmdArg::none,const RooCmdArg& arg6=RooCmdArg::none,
                 const RooCmdArg& arg7=RooCmdArg::none,const RooCmdArg& arg8=RooCmdArg::none) ;
RooCmdArg Sibling(const RooAbsCollection& sibling) ;

// RooAbsRealLValue::frame arguments
RooCmdArg Title(const char* name) ;
RooCmdArg Bins(Int_t nbin) ;
RooCmdArg AutoSymRange(const RooAbsData& data, Double_t marginFactor=0.1) ;
RooCmdArg AutoRange(const RooAbsData& data, Double_t marginFactor=0.1) ;

// RooAbsData::reduce arguments
RooCmdArg SelectVars(const RooArgSet& vars) ;
RooCmdArg EventRange(Int_t nStart, Int_t nStop) ;

// RooAbsPdf::fitTo arguments
RooCmdArg FitOptions(const char* opts) ;
RooCmdArg Optimize(Bool_t flag=kTRUE) ;
RooCmdArg ProjectedObservables(const RooArgSet& set) ; // obsolete, for backward compatibility
RooCmdArg ConditionalObservables(const RooArgSet& set) ;
RooCmdArg Verbose(Bool_t flag=kTRUE) ;
RooCmdArg Save(Bool_t flag=kTRUE) ;
RooCmdArg Timer(Bool_t flag=kTRUE) ;
RooCmdArg PrintLevel(Int_t code) ;
RooCmdArg Strategy(Int_t code) ;
RooCmdArg InitialHesse(Bool_t flag=kTRUE) ;
RooCmdArg Hesse(Bool_t flag=kTRUE) ;
RooCmdArg Minos(Bool_t flag=kTRUE) ;
RooCmdArg Minos(const RooArgSet& minosArgs) ;
RooCmdArg SplitRange(Bool_t flag=kTRUE) ;

// RooAbsPdf::paramOn arguments
RooCmdArg Label(const char* str) ;
RooCmdArg Layout(Double_t xmin, Double_t xmax=0.99, Double_t ymin=0.95) ;
RooCmdArg Parameters(const RooArgSet& params) ;
RooCmdArg ShowConstants(Bool_t flag=kTRUE) ;

// RooTreeData::statOn arguments
RooCmdArg What(const char* str) ;

// RooProdPdf::ctor arguments
RooCmdArg Conditional(const RooArgSet& pdfSet, const RooArgSet& depSet) ;

// RooAbsPdf::generate arguments
RooCmdArg ProtoData(const RooDataSet& protoData, Bool_t randomizeOrder=kFALSE, Bool_t resample=kFALSE) ;
RooCmdArg NumEvents(Int_t numEvents) ;

// RooAbsRealLValue::createHistogram arguments
RooCmdArg YVar(const RooAbsRealLValue& var, const RooCmdArg& arg=RooCmdArg::none) ;
RooCmdArg ZVar(const RooAbsRealLValue& var, const RooCmdArg& arg=RooCmdArg::none) ;
RooCmdArg AxisLabel(const char* name) ;

// RooAbsReal::createIntegral arguments
RooCmdArg NormSet(const RooArgSet& nset) ;
RooCmdArg NumIntConfig(const RooNumIntConfig& cfg) ;

// RooMCStudy::ctor arguments
RooCmdArg FitModel(RooAbsPdf& pdf) ;
RooCmdArg FitOptions(const RooCmdArg& arg1                ,const RooCmdArg& arg2=RooCmdArg::none,
                     const RooCmdArg& arg3=RooCmdArg::none,const RooCmdArg& arg4=RooCmdArg::none,
                     const RooCmdArg& arg5=RooCmdArg::none,const RooCmdArg& arg6=RooCmdArg::none) ;
RooCmdArg Binned(Bool_t flag=kTRUE) ;

// RooMCStudy::plot* arguments
RooCmdArg Frame(const RooCmdArg& arg1                ,const RooCmdArg& arg2=RooCmdArg::none,
                const RooCmdArg& arg3=RooCmdArg::none,const RooCmdArg& arg4=RooCmdArg::none,
                const RooCmdArg& arg5=RooCmdArg::none,const RooCmdArg& arg6=RooCmdArg::none) ;
RooCmdArg FrameBins(Int_t nbins) ;
RooCmdArg FrameRange(Double_t xlo, Double_t xhi) ;
RooCmdArg FitGauss(Bool_t flag=kTRUE) ;

// RooRealVar::format arguments
RooCmdArg AutoPrecision(Int_t ndigit=2) ;
RooCmdArg FixedPrecision(Int_t ndigit=2) ;
RooCmdArg TLatexStyle(Bool_t flag=kTRUE) ;
RooCmdArg LatexStyle(Bool_t flag=kTRUE) ;
RooCmdArg LatexTableStyle(Bool_t flag=kTRUE) ;
RooCmdArg VerbatimName(Bool_t flag=kTRUE) ;

RooConstVar& RooConst(Double_t val) ; 

}

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

} // End namespace ShortHand

#endif
