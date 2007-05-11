/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsData.cc,v 1.29 2006/12/08 15:50:40 wverkerke Exp $
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

// -- CLASS DESCRIPTION [DATA] --
// RooAbsData is the common abstract base class for binned and unbinned
// datasets. The abstract interface defines plotting and tabulating entry
// points for its contents and provides an iterator over its elements
// (bins for binned data sets, data points for unbinned datasets).

#include "RooFit.h"

#include "RooAbsData.h"
#include "RooAbsData.h"
#include "RooFormulaVar.h"
#include "RooCmdConfig.h"
#include "RooAbsRealLValue.h"


ClassImp(RooAbsData)
;


RooAbsData::RooAbsData() 
{
  // Default constructor
  _iterator = _vars.createIterator() ;
  _cacheIter = _cachedVars.createIterator() ;
}


RooAbsData::RooAbsData(const char *name, const char *title, const RooArgSet& vars) :
  TNamed(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), 
  _doDirtyProp(kTRUE) 
{
  // Constructor from a set of variables. Only fundamental elements of vars
  // (RooRealVar,RooCategory etc) are stored as part of the dataset

  // clone the fundamentals of the given data set into internal buffer
  TIterator* iter = vars.createIterator() ;
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {
    if (!var->isFundamental()) {
      cout << "RooAbsSet::initialize(" << GetName() 
	   << "): Data set cannot contain non-fundamental types, ignoring " 
	   << var->GetName() << endl ;
    } else {
      _vars.addClone(*var);
    }
  }
  delete iter ;

  _iterator= _vars.createIterator();
  _cacheIter = _cachedVars.createIterator() ;
}




RooAbsData::RooAbsData(const RooAbsData& other, const char* newname) : 
  TNamed(newname?newname:other.GetName(),other.GetTitle()), 
  RooPrintable(other), _vars(),
  _cachedVars("Cached Variables"), _doDirtyProp(kTRUE)
{
  // Copy constructor
  _vars.addClone(other._vars) ;
  _iterator= _vars.createIterator();
  _cacheIter = _cachedVars.createIterator() ;
}


RooAbsData::~RooAbsData() 
{
  // Destructor
  
  // delete owned contents.
  delete _iterator ;
  delete _cacheIter ;
}



RooAbsData* RooAbsData::reduce(RooCmdArg arg1,RooCmdArg arg2,RooCmdArg arg3,RooCmdArg arg4,
			       RooCmdArg arg5,RooCmdArg arg6,RooCmdArg arg7,RooCmdArg arg8) 
{
  // Create a reduced copy of this dataset. The caller takes ownership of the returned dataset
  //
  // The following optional named arguments are accepted
  //
  //   SelectVars(const RooArgSet& vars) -- Only retain the listed observables in the output dataset
  //   Cut(const char* expression)       -- Only retain event surviving the given cut expression
  //   Cut(const RooFormulaVar& expr)    -- Only retain event surviving the given cut formula
  //   CutRange(const char* name)        -- Only retain events inside range with given name. Multiple CutRange
  //                                        arguments may be given to select multiple ranges
  //   EventRange(int lo, int hi)        -- Only retain events with given sequential event numbers
  //   Name(const char* name)            -- Give specified name to output dataset
  //   Title(const char* name)           -- Give specified title to output dataset
  //

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsData::reduce(%s)",GetName())) ;
  pc.defineString("name","Name",0,"") ;
  pc.defineString("title","Title",0,"") ;
  pc.defineString("cutRange","CutRange",0,"") ;
  pc.defineString("cutSpec","CutSpec",0,"") ;
  pc.defineObject("cutVar","CutVar",0,0) ;
  pc.defineInt("evtStart","EventRange",0,0) ;
  pc.defineInt("evtStop","EventRange",1,2000000000) ;
  pc.defineObject("varSel","SelectVars",0,0) ;
  pc.defineMutex("CutVar","CutSpec") ;

  // Process & check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Extract values from named arguments
  const char* cutRange = pc.getString("cutRange",0,kTRUE) ;
  const char* cutSpec = pc.getString("cutSpec",0,kTRUE) ;
  RooFormulaVar* cutVar = static_cast<RooFormulaVar*>(pc.getObject("cutVar",0)) ;
  Int_t nStart = pc.getInt("evtStart",0) ;
  Int_t nStop = pc.getInt("evtStop",2000000000) ;
  RooArgSet* varSet = static_cast<RooArgSet*>(pc.getObject("varSel")) ;
  const char* name = pc.getString("name",0,kTRUE) ;
  const char* title = pc.getString("title",0,kTRUE) ;

  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset ;
  if (varSet) {
    varSubset.add(*varSet) ;
    TIterator* iter = varSubset.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (!_vars.find(arg->GetName())) {
	cout << "RooAbsData::reduce(" << GetName() << ") WARNING: variable " 
	     << arg->GetName() << " not in dataset, ignored" << endl ;
	varSubset.remove(*arg) ;
      }
    }
    delete iter ;    
  } else {
    varSubset.add(*get()) ;
  }

  RooAbsData* ret = 0 ;
  if (cutSpec) {

    RooFormulaVar cutVarTmp(cutSpec,cutSpec,*get()) ;
    ret =  reduceEng(varSubset,&cutVarTmp,cutRange,nStart,nStop,kFALSE) ;      

  } else if (cutVar) {

    ret = reduceEng(varSubset,cutVar,cutRange,nStart,nStop,kFALSE) ;

  } else {

    ret = reduceEng(varSubset,0,cutRange,nStart,nStop,kFALSE) ;

  }
  
  if (!ret) return 0 ;

  if (name) {
    ret->SetName(name) ;
  }
  if (title) {
    ret->SetTitle(title) ;
  }

  return ret ;
}


RooAbsData* RooAbsData::reduce(const char* cut) 
{ 
  // Create a subset of the data set by applying the given cut on the data points.
  // The cut expression can refer to any variable in the data set. For cuts involving 
  // other variables, such as intermediate formula objects, use the equivalent 
  // reduce method specifying the as a RooFormulVar reference.

  RooFormulaVar cutVar(cut,cut,*get()) ;
  return reduceEng(*get(),&cutVar,0,0,2000000000,kFALSE) ;
}




RooAbsData* RooAbsData::reduce(const RooFormulaVar& cutVar) 
{
  // Create a subset of the data set by applying the given cut on the data points.
  // The 'cutVar' formula variable is used to select the subset of data points to be 
  // retained in the reduced data collection.
  return reduceEng(*get(),&cutVar,0,0,2000000000,kFALSE) ;
}




RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const char* cut) 
{
  // Create a subset of the data set by applying the given cut on the data points
  // and reducing the dimensions to the specified set.
  // 
  // The cut expression can refer to any variable in the data set. For cuts involving 
  // other variables, such as intermediate formula objects, use the equivalent 
  // reduce method specifying the as a RooFormulVar reference.

  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset2(varSubset) ;
  TIterator* iter = varSubset.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!_vars.find(arg->GetName())) {
      cout << "RooAbsData::reduce(" << GetName() << ") WARNING: variable " 
	   << arg->GetName() << " not in dataset, ignored" << endl ;
      varSubset2.remove(*arg) ;
    }
  }
  delete iter ;

  if (cut && strlen(cut)>0) {
    RooFormulaVar cutVar(cut,cut,*get()) ;
    return reduceEng(varSubset2,&cutVar,0,0,2000000000,kFALSE) ;      
  } 
  return reduceEng(varSubset2,0,0,0,2000000000,kFALSE) ;
}




RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const RooFormulaVar& cutVar) 
{
  // Create a subset of the data set by applying the given cut on the data points
  // and reducing the dimensions to the specified set.
  // 
  // The 'cutVar' formula variable is used to select the subset of data points to be 
  // retained in the reduced data collection.

  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset2(varSubset) ;
  TIterator* iter = varSubset.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!_vars.find(arg->GetName())) {
      cout << "RooAbsData::reduce(" << GetName() << ") WARNING: variable " 
	   << arg->GetName() << " not in dataset, ignored" << endl ;
      varSubset2.remove(*arg) ;
    }
  }
  delete iter ;

  return reduceEng(varSubset2,&cutVar,0,0,2000000000,kFALSE) ;
}


Double_t RooAbsData::weightError(ErrorType) const 
{ 
  return 0 ; 
} 

void RooAbsData::weightError(Double_t& lo, Double_t& hi, ErrorType) const 
{ 
  lo=0 ; hi=0 ; 
} 


RooPlot* RooAbsData::plotOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
			    const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5, 
			    const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
  // Plot dataset on specified frame. By default an unbinned dataset will use the default binning of
  // the target frame. A binned dataset will by default retain its intrinsic binning.
  //
  // The following optional named arguments can be used to modify the default behavior
  //
  // Data representation options
  // ---------------------------
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the daya in given two-state category [F(+)-F(-)] / [F(+)+F(-)]. 
  //                                    Category must have two states with indices -1 and +1 or three states with indeces -1,0 and +1.
  // ErrorType(RooAbsData::EType)    -- Select the type of error drawn: Poisson (default) draws asymmetric Poisson
  //                                    confidence intervals. SumW2 draws symmetric sum-of-weights error
  // Binning(double xlo, double xhi, -- Use specified binning to draw dataset
  //                      int nbins)
  // Binning(const RooAbsBinning&)   -- Use specified binning to draw dataset
  // Binning(const char* name)       -- Use binning with specified name to draw dataset
  // RefreshNorm(Bool_t flag)        -- Force refreshing for PDF normalization information in frame.
  //                                    If set, any subsequent PDF will normalize to this dataset, even if it is
  //                                    not the first one added to the frame. By default only the 1st dataset
  //                                    added to a frame will update the normalization information
  //
  // Histogram drawing options
  // -------------------------
  // DrawOption(const char* opt)     -- Select ROOT draw option for resulting TGraph object
  // LineStyle(Int_t style)          -- Select line style by ROOT line style code, default is solid
  // LineColor(Int_t color)          -- Select line color by ROOT color code, default is black
  // LineWidth(Int_t width)          -- Select line with in pixels, default is 3
  // MarkerStyle(Int_t style)        -- Select the ROOT marker style, default is 21
  // MarkerColor(Int_t color)        -- Select the ROOT marker color, default is black
  // MarkerSize(Double_t size)       -- Select the ROOT marker size
  // XErrorSize(Double_t frac)       -- Select size of X error bar as fraction of the bin width, default is 1
  //
  //
  // Misc. other options
  // -------------------
  // Name(const chat* name)          -- Give curve specified name in frame. Useful if curve is to be referenced later
  // Invisble(Bool_t flag)           -- Add curve to frame, but do not display. Useful in combination AddTo()
  // AddTo(const char* name,         -- Add constructed histogram to already existing histogram with given name and relative weight factors
  // double_t wgtSelf, double_t wgtOther)
  // 
  //                                    
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return plotOn(frame,l) ;  
}

TH1 *RooAbsData::createHistogram(const char *name, const RooAbsRealLValue& xvar,
				 const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
				 const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const 
{
  // Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this dataset. 
  //
  // This function accepts the following arguments
  //
  // name -- Name of the ROOT histogram
  // xvar -- Observable to be mapped on x axis of ROOT histogram
  //
  // Binning(const char* name)                    -- Apply binning with given name to x axis of histogram
  // Binning(RooAbsBinning& binning)              -- Apply specified binning to x axis of histogram
  // Binning(int nbins, double lo, double hi)     -- Apply specified binning to x axis of histogram
  //
  // YVar(const RooAbsRealLValue& var,...)    -- Observable to be mapped on y axis of ROOT histogram
  // ZVar(const RooAbsRealLValue& var,...)    -- Observable to be mapped on z axis of ROOT histogram
  //
  // The YVar() and ZVar() arguments can be supplied with optional Binning() arguments to control the binning of the Y and Z axes, e.g.
  // createHistogram("histo",x,Binning(-1,1,20), YVar(y,Binning(-1,1,30)), ZVar(z,Binning("zbinning")))
  //
  // The caller takes ownership of the returned histogram

  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsData::createHistogram(%s)",GetName())) ;
  pc.defineString("cutRange","CutRange",0,"",kTRUE) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineObject("yvar","YVar",0,0) ;
  pc.defineObject("zvar","ZVar",0,0) ;
  pc.allowUndefined() ;
  
  // Process & check varargs 
  pc.process(l) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  const char* cutSpec = pc.getString("cutString",0,kTRUE) ;
  const char* cutRange = pc.getString("cutRange",0,kTRUE) ;

  RooArgList vars(xvar) ;
  RooAbsArg* yvar = static_cast<RooAbsArg*>(pc.getObject("yvar")) ;
  if (yvar) {
    vars.add(*yvar) ;
  }
  RooAbsArg* zvar = static_cast<RooAbsArg*>(pc.getObject("zvar")) ;
  if (zvar) {
    vars.add(*zvar) ;
  }

  pc.stripCmdList(l,"CutRange,CutSpec") ;
  TH1* histo = xvar.createHistogram(name,l) ;
  fillHistogram(histo,vars,cutSpec,cutRange) ;

  return histo ;
}
