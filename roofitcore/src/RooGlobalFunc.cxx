/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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

// Global helper functions

#include "RooFitCore/RooGlobalFunc.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooRealConstant.hh"

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
RooCmdArg Range(Double_t lo, Double_t hi, Bool_t vlines) { return RooCmdArg("Range",vlines?1:0,0,lo,hi,0,0,0,0) ; }
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
RooCmdArg Cut(const RooAbsReal& cutVar)         { return RooCmdArg("CutVar",0,0,0,0,0,0,&cutVar,0) ; }
RooCmdArg Binning(const RooAbsBinning& binning) { return RooCmdArg("Binning",0,0,0,0,0,0,&binning,0) ;}
RooCmdArg MarkerStyle(Style_t style)            { return RooCmdArg("MarkerStyle",style,0,0,0,0,0,0,0) ; }
RooCmdArg MarkerSize(Size_t size)               { return RooCmdArg("MarkerSize",0,0,size,0,0,0,0,0) ; }
RooCmdArg MarkerColor(Color_t color)            { return RooCmdArg("MarkerColor",color,0,0,0,0,0,0,0) ; }
RooCmdArg CutRange(const char* rangeName)       { return RooCmdArg("CutRange",0,0,0,0,rangeName,0,0,0) ; }
RooCmdArg AddTo(const char* name)               { return RooCmdArg("AddTo",0,0,0,0,name,0,0,0) ; }
RooCmdArg XErrorSize(Double_t width)            { return RooCmdArg("XErrorSize",0,0,width,0,0,0,0,0) ; }
RooCmdArg RefreshNorm()                         { return RooCmdArg("RefreshNorm",1,0,0,0,0,0,0,0) ; }

// RooChi2Var::ctor arguments
RooCmdArg Extended() { return RooCmdArg("Extended",1,0,0,0,0,0,0,0) ; }
RooCmdArg DataError(RooDataHist::ErrorType etype) { return RooCmdArg("DataError",(Int_t)etype,0,0,0,0,0,0,0) ; }
RooCmdArg NumCPU(Int_t nCPU) { return RooCmdArg("NumCPU",nCPU,0,0,0,0,0,0,0) ; }

// RooAbsCollection::printLatex arguments
RooCmdArg Columns(Int_t ncol)                           { return RooCmdArg("Columns",ncol,0,0,0,0,0,0,0) ; }
RooCmdArg OutputFile(const char* fileName)              { return RooCmdArg("OutputFile",0,0,0,0,fileName,0,0,0) ; }
RooCmdArg Format(const char* format, Int_t sigDigit)    { return RooCmdArg("Format",sigDigit,0,0,0,format,0,0,0) ; }
RooCmdArg Sibling(const RooAbsCollection& sibling)      { return RooCmdArg("Sibling",0,0,0,0,0,0,&sibling,0) ; }

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

RooConstVar& C(Double_t value) { return RooConst(value) ; }

}
