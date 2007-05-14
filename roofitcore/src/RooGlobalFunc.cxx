/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooGlobalFunc.cxx,v 1.13 2007/05/11 09:11:58 verkerke Exp $
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

// Global helper functions

#include "RooFit.h"

#include "RooGlobalFunc.h"
#include "RooGlobalFunc.h"
#include "RooCategory.h"
#include "RooRealConstant.h"
#include "RooDataSet.h"
#include "RooNumIntConfig.h"

namespace RooFit {

  // RooAbsReal::plotOn arguments
  RooCmdArg DrawOption(const char* opt)            { return RooCmdArg("DrawOption",0,0,0,0,opt,0,0,0) ; }
  RooCmdArg Slice(const RooArgSet& sliceSet)       { return RooCmdArg("SliceVars",0,0,0,0,0,0,&sliceSet,0) ; }
  RooCmdArg Project(const RooArgSet& projSet)      { return RooCmdArg("Project",0,0,0,0,0,0,&projSet,0) ; }
  RooCmdArg ProjWData(const RooArgSet& projSet, const RooAbsData& projData) { return RooCmdArg("ProjData",0,0,0,0,0,0,&projSet,&projData) ; }
  RooCmdArg ProjWData(const RooAbsData& projData)  { return RooCmdArg("ProjData",0,0,0,0,0,0,0,&projData) ; }
  RooCmdArg Asymmetry(const RooCategory& cat)      { return RooCmdArg("Asymmetry",0,0,0,0,0,0,&cat,0) ; }
  RooCmdArg Precision(Double_t prec)               { return RooCmdArg("Precision",0,0,prec,0,0,0,0,0) ; }
  RooCmdArg ShiftToZero()                          { return RooCmdArg("ShiftToZero",1,0,0,0,0,0,0,0) ; }
  RooCmdArg Normalization(Double_t scaleFactor)    { return RooCmdArg("Normalization",RooAbsReal::Relative,0,scaleFactor,0,0,0,0,0) ; }
  RooCmdArg Range(const char* rangeName, Bool_t adjustNorm)   { return RooCmdArg("RangeWithName",adjustNorm,0,0,0,rangeName,0,0,0) ; }
  RooCmdArg Range(Double_t lo, Double_t hi, Bool_t adjustNorm){ return RooCmdArg("Range",adjustNorm,0,lo,hi,0,0,0,0) ; }
  RooCmdArg VLines()                               { return RooCmdArg("VLines",1,0,0,0,0,0,0,0) ; } 
  RooCmdArg LineColor(Color_t color)               { return RooCmdArg("LineColor",color,0,0,0,0,0,0,0) ; }
  RooCmdArg LineStyle(Style_t style)               { return RooCmdArg("LineStyle",style,0,0,0,0,0,0,0) ; }
  RooCmdArg LineWidth(Width_t width)               { return RooCmdArg("LineWidth",width,0,0,0,0,0,0,0) ; }
  RooCmdArg FillColor(Color_t color)               { return RooCmdArg("FillColor",color,0,0,0,0,0,0,0) ; }
  RooCmdArg FillStyle(Style_t style)               { return RooCmdArg("FillStyle",style,0,0,0,0,0,0,0) ; }
  RooCmdArg ProjectionRange(const char* rangeName) { return RooCmdArg("ProjectionRange",0,0,0,0,rangeName,0,0,0) ; }
  RooCmdArg Name(const char* name)                 { return RooCmdArg("Name",0,0,0,0,name,0,0,0) ; }
  RooCmdArg Invisible()                            { return RooCmdArg("Invisible",1,0,0,0,0,0,0,0) ; }
  RooCmdArg AddTo(const char* name, double wgtSel, double wgtOther) { return RooCmdArg("AddTo",0,0,wgtSel,wgtOther,name,0,0,0) ; }
  
  // RooAbsPdf::plotOn arguments
  RooCmdArg Components(const RooArgSet& compSet) { return RooCmdArg("SelectCompSet",0,0,0,0,0,0,&compSet,0) ; }
  RooCmdArg Components(const char* compSpec) { return RooCmdArg("SelectCompSpec",0,0,0,0,compSpec,0,0,0) ; }
  RooCmdArg Normalization(Double_t scaleFactor, RooAbsPdf::ScaleType scaleType) 
                                                   { return RooCmdArg("Normalization",scaleType,0,scaleFactor,0,0,0,0,0) ; }
  
  // RooAbsData::plotOn arguments
  RooCmdArg Cut(const char* cutSpec)              { return RooCmdArg("CutSpec",0,0,0,0,cutSpec,0,0,0) ; }
  RooCmdArg Cut(const RooFormulaVar& cutVar)      { return RooCmdArg("CutVar",0,0,0,0,0,0,&cutVar,0) ; }
  RooCmdArg Binning(const RooAbsBinning& binning) { return RooCmdArg("Binning",0,0,0,0,0,0,&binning,0) ;}
  RooCmdArg Binning(const char* binningName) { return RooCmdArg("BinningName",0,0,0,0,binningName,0,0,0) ;}
  RooCmdArg Binning(Int_t nBins, Double_t xlo, Double_t xhi) { return RooCmdArg("BinningSpec",nBins,0,xlo,xhi,0,0,0,0) ;}
  RooCmdArg MarkerStyle(Style_t style)            { return RooCmdArg("MarkerStyle",style,0,0,0,0,0,0,0) ; }
  RooCmdArg MarkerSize(Size_t size)               { return RooCmdArg("MarkerSize",0,0,size,0,0,0,0,0) ; }
  RooCmdArg MarkerColor(Color_t color)            { return RooCmdArg("MarkerColor",color,0,0,0,0,0,0,0) ; }
  RooCmdArg CutRange(const char* rangeName)       { return RooCmdArg("CutRange",0,0,0,0,rangeName,0,0,0) ; }
  RooCmdArg AddTo(const char* name)               { return RooCmdArg("AddTo",0,0,0,0,name,0,0,0) ; }
  RooCmdArg XErrorSize(Double_t width)            { return RooCmdArg("XErrorSize",0,0,width,0,0,0,0,0) ; }
  RooCmdArg RefreshNorm()                         { return RooCmdArg("RefreshNorm",1,0,0,0,0,0,0,0) ; }
  
  // RooChi2Var::ctor arguments
  RooCmdArg Extended(Bool_t flag) { return RooCmdArg("Extended",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg DataError(RooDataHist::ErrorType etype) { return RooCmdArg("DataError",(Int_t)etype,0,0,0,0,0,0,0) ; }
  RooCmdArg NumCPU(Int_t nCPU) { return RooCmdArg("NumCPU",nCPU,0,0,0,0,0,0,0) ; }
  
  // RooAbsCollection::printLatex arguments
  RooCmdArg Columns(Int_t ncol)                           { return RooCmdArg("Columns",ncol,0,0,0,0,0,0,0) ; }
  RooCmdArg OutputFile(const char* fileName)              { return RooCmdArg("OutputFile",0,0,0,0,fileName,0,0,0) ; }
  RooCmdArg Sibling(const RooAbsCollection& sibling)      { return RooCmdArg("Sibling",0,0,0,0,0,0,&sibling,0) ; }
  RooCmdArg Format(const char* format, Int_t sigDigit)    { return RooCmdArg("Format",sigDigit,0,0,0,format,0,0,0) ; }
  RooCmdArg Format(const char* what, const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,const RooCmdArg& arg4,
                   const RooCmdArg& arg5,const RooCmdArg& arg6,const RooCmdArg& arg7,const RooCmdArg& arg8) {
            RooCmdArg ret("FormatArgs",0,0,0,0,what,0,0,0) ; ret.addArg(arg1) ; ret.addArg(arg2) ; 
            ret.addArg(arg3) ; ret.addArg(arg4) ; ret.addArg(arg5) ; ret.addArg(arg6) ; ret.addArg(arg7) ; ret.addArg(arg8) ;
            ret.setProcessRecArgs(kFALSE) ; return ret ;
  }
  
  // RooAbsRealLValue::frame arguments
  RooCmdArg Title(const char* name) { return RooCmdArg("Title",0,0,0,0,name,0,0,0) ; }
  RooCmdArg Bins(Int_t nbin)        { return RooCmdArg("Bins",nbin,0,0,0,0,0,0,0) ; }
  RooCmdArg AutoSymRange(const RooAbsData& data, Double_t marginFactor) { return RooCmdArg("AutoRange",1,0,marginFactor,0,0,0,&data,0) ; }
  RooCmdArg AutoRange(const RooAbsData& data, Double_t marginFactor) { return RooCmdArg("AutoRange",0,0,marginFactor,0,0,0,&data,0) ; }
  
  // RooAbsData::reduce arguments
  RooCmdArg SelectVars(const RooArgSet& vars)     { return RooCmdArg("SelectVars",0,0,0,0,0,0,&vars,0) ; }
  RooCmdArg EventRange(Int_t nStart, Int_t nStop) { return RooCmdArg("EventRange",nStart,nStop,0,0,0,0,0,0) ; }
  
  // RooAbsPdf::fitTo arguments
  RooCmdArg FitOptions(const char* opts) { return RooCmdArg("FitOptions",0,0,0,0,opts,0,0,0) ; }
  RooCmdArg Optimize(Bool_t flag)        { return RooCmdArg("Optimize",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg Verbose(Bool_t flag)         { return RooCmdArg("Verbose",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg Save(Bool_t flag)            { return RooCmdArg("Save",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg Timer(Bool_t flag)           { return RooCmdArg("Timer",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg PrintLevel(Int_t level)      { return RooCmdArg("PrintLevel",level,0,0,0,0,0,0,0) ; }
  RooCmdArg Strategy(Int_t code)         { return RooCmdArg("Strategy",code,0,0,0,0,0,0,0) ; }
  RooCmdArg InitialHesse(Bool_t flag)    { return RooCmdArg("InitialHesse",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg Hesse(Bool_t flag)           { return RooCmdArg("Hesse",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg Minos(Bool_t flag)           { return RooCmdArg("Minos",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg Minos(const RooArgSet& minosArgs)            { return RooCmdArg("Minos",kTRUE,0,0,0,0,0,&minosArgs,0) ; }
  RooCmdArg ConditionalObservables(const RooArgSet& set) { return RooCmdArg("ProjectedObservables",0,0,0,0,0,0,&set,0) ; }
  RooCmdArg ProjectedObservables(const RooArgSet& set)   { return RooCmdArg("ProjectedObservables",0,0,0,0,0,0,&set,0) ; }
  RooCmdArg SplitRange(Bool_t flag)                      { return RooCmdArg("SplitRange",flag,0,0,0,0,0,0,0) ; }
  
  // RooAbsPdf::paramOn arguments
  RooCmdArg Label(const char* str) { return RooCmdArg("Label",0,0,0,0,str,0,0,0) ; }
  RooCmdArg Layout(Double_t xmin, Double_t xmax, Double_t ymin) { return RooCmdArg("Layout",Int_t(ymin*10000),0,xmin,xmax,0,0,0,0) ; }
  RooCmdArg Parameters(const RooArgSet& params) { return RooCmdArg("Parameters",0,0,0,0,0,0,&params,0) ; }
  RooCmdArg ShowConstants(Bool_t flag) { return RooCmdArg("ShowConstants",flag,0,0,0,0,0,0,0) ; }

  // RooTreeData::statOn arguments
  RooCmdArg What(const char* str) { return RooCmdArg("What",0,0,0,0,str,0,0,0) ; }

  // RooProdPdf::ctor arguments
  RooCmdArg Conditional(const RooArgSet& pdfSet, const RooArgSet& depSet) { return RooCmdArg("Conditional",0,0,0,0,0,0,&pdfSet,&depSet) ; } ;
  
  // RooAbsPdf::generate arguments
  RooCmdArg ProtoData(const RooDataSet& protoData, Bool_t randomizeOrder, Bool_t resample) 
                                         { return RooCmdArg("PrototypeData",randomizeOrder,resample,0,0,0,0,&protoData,0) ; }
  RooCmdArg NumEvents(Int_t numEvents)   { return RooCmdArg("NumEvents",numEvents,0,0,0,0,0,0,0) ; }
  
  // RooAbsRealLValue::createHistogram arguments
  RooCmdArg YVar(const RooAbsRealLValue& var, const RooCmdArg& arg)       { return RooCmdArg("YVar",0,0,0,0,0,0,&var,0,&arg) ; }
  RooCmdArg ZVar(const RooAbsRealLValue& var, const RooCmdArg& arg)       { return RooCmdArg("ZVar",0,0,0,0,0,0,&var,0,&arg) ; }
  RooCmdArg AxisLabel(const char* name)                                   { return RooCmdArg("AxisLabel",0,0,0,0,name,0,0,0) ; }

  // RooAbsReal::createIntegral arguments
  RooCmdArg NormSet(const RooArgSet& nset)           { return RooCmdArg("NormSet",0,0,0,0,0,0,&nset,0) ; }
  RooCmdArg NumIntConfig(const RooNumIntConfig& cfg) { return RooCmdArg("NumIntConfig",0,0,0,0,0,0,&cfg,0) ; }

  // RooMCStudy::ctor arguments
  RooCmdArg FitModel(RooAbsPdf& pdf) { return RooCmdArg("FitModel",0,0,0,0,0,0,&pdf,0) ; }
  RooCmdArg FitOptions(const RooCmdArg& arg1 ,const RooCmdArg& arg2, const RooCmdArg& arg3,
                       const RooCmdArg& arg4, const RooCmdArg& arg5, const RooCmdArg& arg6) {
             RooCmdArg ret("FitOptArgs",0,0,0,0,0,0,0,0) ; ret.addArg(arg1) ; ret.addArg(arg2) ; 
             ret.addArg(arg3) ; ret.addArg(arg4) ; ret.addArg(arg5) ; ret.addArg(arg6) ; 
             ret.setProcessRecArgs(kFALSE) ; return ret ; 
  }
  RooCmdArg Binned(Bool_t flag)               { return RooCmdArg("Binned",flag,0,0,0,0,0,0,0) ; }

  // RooMCStudy::plot* arguments
  RooCmdArg Frame(const RooCmdArg& arg1,const RooCmdArg& arg2,
                  const RooCmdArg& arg3,const RooCmdArg& arg4,
                  const RooCmdArg& arg5,const RooCmdArg& arg6) {
            RooCmdArg ret("FrameArgs",0,0,0,0,0,0,0,0) ; ret.addArg(arg1) ; ret.addArg(arg2) ; 
            ret.addArg(arg3) ; ret.addArg(arg4) ; ret.addArg(arg5) ; ret.addArg(arg6) ; 
            ret.setProcessRecArgs(kFALSE) ; return ret ;
  }
  RooCmdArg FrameBins(Int_t nbins)                 { return RooCmdArg("FrameBins",nbins,0,0,0,0,0,0,0) ; }
  RooCmdArg FrameRange(Double_t xlo, Double_t xhi) { return RooCmdArg("FrameRange",0,0,xlo,xhi,0,0,0,0) ; }
  RooCmdArg FitGauss(Bool_t flag)                  { return RooCmdArg("FitGauss",flag,0,0,0,0,0,0,0) ; }

  // RooRealVar::format arguments
  RooCmdArg ShowName(Bool_t flag)             { return RooCmdArg("ShowName",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg ShowValue(Bool_t flag)            { return RooCmdArg("ShowValue",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg ShowError(Bool_t flag)            { return RooCmdArg("ShowError",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg ShowAsymError(Bool_t flag)        { return RooCmdArg("ShowAsymError",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg ShowUnit(Bool_t flag)             { return RooCmdArg("ShowUnit",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg AutoPrecision(Int_t ndigit)   { return RooCmdArg("AutoPrecision",ndigit,0,0,0,0,0,0,0) ; }
  RooCmdArg FixedPrecision(Int_t ndigit)  { return RooCmdArg("FixedPrecision",ndigit,0,0,0,0,0,0,0) ; }
  RooCmdArg TLatexStyle(Bool_t flag)      { return RooCmdArg("TLatexStyle",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg LatexStyle(Bool_t flag)       { return RooCmdArg("LatexStyle",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg LatexTableStyle(Bool_t flag)  { return RooCmdArg("LatexTableStyle",flag,0,0,0,0,0,0,0) ; }
  RooCmdArg VerbatimName(Bool_t flag)     { return RooCmdArg("VerbatimName",flag,0,0,0,0,0,0,0) ; }

  RooConstVar& RooConst(Double_t val) { return RooRealConstant::value(val) ; }
 
} // End namespace RooFit

namespace RooFitShortHand {

RooArgSet S(const RooAbsArg& v1) { return RooArgSet(v1) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2) { return RooArgSet(v1,v2) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3) { return RooArgSet(v1,v2,v3) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4) { return RooArgSet(v1,v2,v3,v4) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5) 
          { return RooArgSet(v1,v2,v3,v4,v5) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6) { return RooArgSet(v1,v2,v3,v4,v5,v6) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6, const RooAbsArg& v7) { return RooArgSet(v1,v2,v3,v4,v5,v6,v7) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8) { return RooArgSet(v1,v2,v3,v4,v5,v6,v7,v8) ; }
RooArgSet S(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
            const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8, const RooAbsArg& v9) 
          { return RooArgSet(v1,v2,v3,v4,v5,v6,v7,v8,v9) ; }

RooArgList L(const RooAbsArg& v1) { return RooArgList(v1) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2) { return RooArgList(v1,v2) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3) { return RooArgList(v1,v2,v3) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4) { return RooArgList(v1,v2,v3,v4) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5) 
           { return RooArgList(v1,v2,v3,v4,v5) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6) { return RooArgList(v1,v2,v3,v4,v5,v6) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6, const RooAbsArg& v7) { return RooArgList(v1,v2,v3,v4,v5,v6,v7) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8) { return RooArgList(v1,v2,v3,v4,v5,v6,v7,v8) ; }
RooArgList L(const RooAbsArg& v1, const RooAbsArg& v2, const RooAbsArg& v3, const RooAbsArg& v4, const RooAbsArg& v5, 
             const RooAbsArg& v6, const RooAbsArg& v7, const RooAbsArg& v8, const RooAbsArg& v9) 
           { return RooArgList(v1,v2,v3,v4,v5,v6,v7,v8,v9) ; }

RooConstVar& C(Double_t value) { return RooFit::RooConst(value) ; }

} // End namespace Shorthand

