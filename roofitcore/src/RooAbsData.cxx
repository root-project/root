/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsData.cc,v 1.21 2004/11/29 20:22:04 wverkerke Exp $
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

// -- CLASS DESCRIPTION [DATA] --
// RooAbsData is the common abstract base class for binned and unbinned
// datasets. The abstract interface defines plotting and tabulating entry
// points for its contents and provides an iterator over its elements
// (bins for binned data sets, data points for unbinned datasets).

#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooCmdConfig.hh"

using std::cout;
using std::endl;

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
  while(0 != (var= (RooAbsArg*)iter->Next())) {
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
  TNamed(newname?newname:other.GetName(),other.GetTitle()), _vars(),
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
    while(arg=(RooAbsArg*)iter->Next()) {
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
  while(arg=(RooAbsArg*)iter->Next()) {
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
  while(arg=(RooAbsArg*)iter->Next()) {
    if (!_vars.find(arg->GetName())) {
      cout << "RooAbsData::reduce(" << GetName() << ") WARNING: variable " 
	   << arg->GetName() << " not in dataset, ignored" << endl ;
      varSubset2.remove(*arg) ;
    }
  }
  delete iter ;

  return reduceEng(varSubset2,&cutVar,0,0,2000000000,kFALSE) ;
}

