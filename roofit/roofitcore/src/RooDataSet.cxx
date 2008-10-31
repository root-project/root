/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooDataSet is a container class to hold unbinned data. Each data point
// in N-dimensional space is represented by a RooArgSet of RooRealVar, RooCategory 
// or RooStringVar objects 
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <fstream>
#include "TTree.h"
#include "TH2.h"
#include "TDirectory.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooAbsReal.h"
#include "Roo1DTable.h"
#include "RooCategory.h"
#include "RooFormulaVar.h"
#include "RooArgList.h"
#include "RooAbsRealLValue.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooMsgService.h"
#include "RooCmdConfig.h"
#include "TROOT.h"

#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

ClassImp(RooDataSet)
;


//_____________________________________________________________________________
RooDataSet::RooDataSet() : _wgtVar(0) 
{
  // Default constructor for persistence
}





//_____________________________________________________________________________
RooDataSet::RooDataSet(const char* name, const char* title, const RooArgSet& vars, RooCmdArg arg1, RooCmdArg arg2, RooCmdArg arg3,
		       RooCmdArg arg4,RooCmdArg arg5,RooCmdArg arg6,RooCmdArg arg7,RooCmdArg arg8)  :
  RooTreeData(name,title,RooArgSet(vars,(RooAbsArg*)RooCmdConfig::decodeObjOnTheFly("RooDataSet::RooDataSet", "IndexCat",0,0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8)))
{
  // Construct an unbinned dataset from a RooArgSet defining the dimensions of the data space. Optionally, data
  // can be imported at the time of construction.
  //
  // This constructor takes the following optional arguments
  //
  // Import(TTree*)              -- Import contents of given TTree. Only braches of the TTree that have names
  //                                corresponding to those of the RooAbsArgs that define the RooDataSet are
  //                                imported. 
  //
  // Import(RooDataSet*)         -- Import contents of given RooDataSet. Only observables that are common with
  //                                the definition of this dataset will be imported
  //
  // Index(RooCategory&)         -- Prepare import of datasets into a N+1 dimensional RooDataSet
  //                                where the extra discrete dimension labels the source of the imported histogram.
  //                              
  // Import(const char*,         -- Import a dataset to be associated with the given state name of the index category
  //              RooDataSet*)      specified in Index(). If the given state name is not yet defined in the index
  //                                category it will be added on the fly. The import command can be specified
  //                                multiple times. 
  //                              
  // Cut(const char*)            -- Apply the given cut specification when importing data
  // Cut(RooFormulaVar&)         
  //
  // CutRange(const char*)       -- Only accept events in the observable range with the given name
  //
  // WeightVar(const char*)      -- Interpret the given variable as event weight rather than as observable
  // WeightVar(const RooAbsArg&) 
  //
  //

  // Define configuration for this method
  RooCmdConfig pc(Form("RooDataSet::ctor(%s)",GetName())) ;
  pc.defineObject("impTree","ImportTree",0) ;
  pc.defineObject("impData","ImportData",0) ;
  pc.defineObject("indexCat","IndexCat",0) ;
  pc.defineObject("impSliceData","ImportDataSlice",0,0,kTRUE) ; // array
  pc.defineString("impSliceState","ImportDataSlice",0,"",kTRUE) ; // array
  pc.defineString("cutSpec","CutSpec",0,"") ; 
  pc.defineObject("cutVar","CutVar",0) ;
  pc.defineString("cutRange","CutRange",0,"") ;
  pc.defineString("wgtVarName","WeightVarName",0,"") ;
  pc.defineObject("wgtVar","WeightVar",0) ;
  pc.defineMutex("ImportTree","ImportData","ImportDataSlice") ;
  pc.defineMutex("CutSpec","CutVar") ;
  pc.defineMutex("WeightVarName","WeightVar") ;
  pc.defineDependency("ImportDataSlice","IndexCat") ;

  
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  // Process & check varargs 
  pc.process(l) ;
  if (!pc.ok(kTRUE)) {
    assert(0) ;
    return ;
  }

  // Extract relevant objects
  TTree* impTree = static_cast<TTree*>(pc.getObject("impTree")) ;
  RooDataSet* impData = static_cast<RooDataSet*>(pc.getObject("impData")) ;
  RooFormulaVar* cutVar = static_cast<RooFormulaVar*>(pc.getObject("cutVar")) ;
  const char* cutSpec = pc.getString("cutSpec","",kTRUE) ;
  const char* cutRange = pc.getString("cutRange","",kTRUE) ;
  const char* wgtVarName = pc.getString("wgtVarName","",kTRUE) ;
  RooRealVar* wgtVar = static_cast<RooRealVar*>(pc.getObject("wgtVar")) ;
  const char* impSliceNames = pc.getString("impSliceState","",kTRUE) ;
  const RooLinkedList& impSliceData = pc.getObjectList("impSliceData") ;
  RooCategory* indexCat = static_cast<RooCategory*>(pc.getObject("indexCat")) ;


  // Make import mapping if index category is specified
  map<string,RooDataSet*> hmap ;  
  if (indexCat) {
    char tmp[1024] ;
    strcpy(tmp,impSliceNames) ;
    char* token = strtok(tmp,",") ;
    TIterator* hiter = impSliceData.MakeIterator() ;
    while(token) {
      hmap[token] = (RooDataSet*) hiter->Next() ;
      token = strtok(0,",") ;
    }
    delete hiter ;
  }

  // Lookup name of weight variable if it was specified by object reference
  if (wgtVar) {
    wgtVarName = wgtVar->GetName() ;
  }


  appendToDir(this,kTRUE) ;

  // Initialize RooDataSet with optional weight variable
  if (wgtVarName && *wgtVarName) {
    // Use the supplied weight column
    initialize(wgtVarName) ;    
  } else {
    if (impData && impData->_wgtVar && vars.find(impData->_wgtVar->GetName())) {
      // Use the weight column of the source data set
      initialize(impData->_wgtVar->GetName()) ;
    } else if (indexCat) {
      RooDataSet* firstDS = hmap.begin()->second ;
      if (firstDS->_wgtVar && vars.find(firstDS->_wgtVar->GetName())) {
	initialize(impData->_wgtVar->GetName()) ;      
      } else {
	initialize(0) ;
      }
    } else {
      initialize(0) ;
    }
  }

  // Import one or more datasets with a cut specification
  if (cutSpec && *cutSpec) {

    // Create a RooFormulaVar cut from given cut expression
    if (indexCat) {

      RooCategory* icat = (RooCategory*) _vars.find(indexCat->GetName()) ;
      for (map<string,RooDataSet*>::iterator hiter = hmap.begin() ; hiter!=hmap.end() ; ++hiter) {
	// Define state labels in index category (both in provided indexCat and in internal copy in dataset)
	if (!indexCat->lookupType(hiter->first.c_str())) {
	  indexCat->defineType(hiter->first.c_str()) ;
	  coutI(InputArguments) << "RooDataSet::ctor(" << GetName() << ") defining state \"" << hiter->first << "\" in index category " << indexCat->GetName() << endl ;
	}
	if (!icat->lookupType(hiter->first.c_str())) {	
	  icat->defineType(hiter->first.c_str()) ;
	}
	icat->setLabel(hiter->first.c_str()) ;

	RooFormulaVar cutVarTmp(cutSpec,cutSpec,hiter->second->_vars) ;
	loadValues(hiter->second,&cutVarTmp,cutRange) ;
      }
      
    } else if (impData) {
      RooFormulaVar cutVarTmp(cutSpec,cutSpec,impData->_vars) ;
      loadValues(impData,&cutVarTmp,cutRange);
    } else if (impTree) {
      RooFormulaVar cutVarTmp(cutSpec,cutSpec,_vars) ;
      loadValues(impTree,&cutVarTmp,cutRange);      
    }

  // Import one or more datasets with a cut formula
  } else if (cutVar) {
    
    if (indexCat) {

      RooCategory* icat = (RooCategory*) _vars.find(indexCat->GetName()) ;
      for (map<string,RooDataSet*>::iterator hiter = hmap.begin() ; hiter!=hmap.end() ; ++hiter) {
	// Define state labels in index category (both in provided indexCat and in internal copy in dataset)
	if (!indexCat->lookupType(hiter->first.c_str())) {
	  indexCat->defineType(hiter->first.c_str()) ;
	  coutI(InputArguments) << "RooDataSet::ctor(" << GetName() << ") defining state \"" << hiter->first << "\" in index category " << indexCat->GetName() << endl ;
	}
	if (!icat->lookupType(hiter->first.c_str())) {	
	  icat->defineType(hiter->first.c_str()) ;
	}
	icat->setLabel(hiter->first.c_str()) ;
	loadValues(hiter->second,cutVar,cutRange) ;
      }


    } else if (impData) {
      loadValues(impData,cutVar,cutRange);
    } else if (impTree) {
      loadValues(impTree,cutVar,cutRange);
    }

  // Import one or more datasets without cuts
  } else {
    
    if (indexCat) {

      RooCategory* icat = (RooCategory*) _vars.find(indexCat->GetName()) ;
      for (map<string,RooDataSet*>::iterator hiter = hmap.begin() ; hiter!=hmap.end() ; ++hiter) {
	// Define state labels in index category (both in provided indexCat and in internal copy in dataset)
	if (!indexCat->lookupType(hiter->first.c_str())) {
	  indexCat->defineType(hiter->first.c_str()) ;
	  coutI(InputArguments) << "RooDataSet::ctor(" << GetName() << ") defining state \"" << hiter->first << "\" in index category " << indexCat->GetName() << endl ;
	}
	if (!icat->lookupType(hiter->first.c_str())) {	
	  icat->defineType(hiter->first.c_str()) ;
	}
	icat->setLabel(hiter->first.c_str()) ;
	loadValues(hiter->second,0,cutRange) ;
      }


    } else if (impData) {
      loadValues(impData,0,cutRange);
    } else if (impTree) {
      loadValues(impTree,0,cutRange);
    }
  }

}





//_____________________________________________________________________________
RooDataSet::RooDataSet(const char *name, const char *title, const RooArgSet& vars, const char* wgtVarName) :
  RooTreeData(name,title,vars)
{
  // Constructor of an empty data set from a RooArgSet defining the dimensions
  // of the data space.

  appendToDir(this,kTRUE) ;
  initialize(wgtVarName) ;
}



//_____________________________________________________________________________
RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *dset, 
		       const RooArgSet& vars, const char *cuts, const char* wgtVarName) :
  RooTreeData(name,title,dset,vars,cuts)
{
  // Constructor of a data set from (part of) an existing data
  // set. The dimensions of the data set are defined by the 'vars'
  // RooArgSet, which can be identical to 'dset' dimensions, or a
  // subset thereof. The 'cuts' string is an optional RooFormula
  // expression and can be used to select the subset of the data
  // points in 'dset' to be copied. The cut expression can refer to
  // any variable in the source dataset. For cuts involving variables
  // other than those contained in the source data set, such as
  // intermediate formula objects, use the equivalent constructor
  // accepting RooFormulaVar reference as cut specification
  //
  // For most uses the RooAbsData::reduce() wrapper function, which
  // uses this constructor, is the most convenient way to create a
  // subset of an existing data
  //
  appendToDir(this,kTRUE) ;

  if (wgtVarName) {
    // Use the supplied weight column
    initialize(wgtVarName) ;    
  } else {
    if (dset->_wgtVar && vars.find(dset->_wgtVar->GetName())) {
      // Use the weight column of the source data set
      initialize(dset->_wgtVar->GetName()) ;
    } else {
      initialize(0) ;
    }
  }

}



//_____________________________________________________________________________
RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *t, 
		       const RooArgSet& vars, const RooFormulaVar& cutVar, const char* wgtVarName) :
  RooTreeData(name,title,t,vars,cutVar)
{
  // Constructor of a data set from (part of) an existing data
  // set. The dimensions of the data set are defined by the 'vars'
  // RooArgSet, which can be identical to 'dset' dimensions, or a
  // subset thereof. The 'cutVar' formula variable is used to select
  // the subset of data points to be copied.  For subsets without
  // selection on the data points, or involving cuts operating
  // exclusively and directly on the data set dimensions, the
  // equivalent constructor with a string based cut expression is
  // recommended.
  //
  // For most uses the RooAbsData::reduce() wrapper function, which
  // uses this constructor, is the most convenient way to create a
  // subset of an existing data

  appendToDir(this,kTRUE) ;

  if (wgtVarName) {
    // Use the supplied weight column
    initialize(wgtVarName) ;    
  } else {
    if (t->_wgtVar && vars.find(t->_wgtVar->GetName())) {
      // Use the weight column of the source data set
      initialize(t->_wgtVar->GetName()) ;
    } else {
      initialize(0) ;
    }
  }
}



//_____________________________________________________________________________
RooDataSet::RooDataSet(const char *name, const char *title, TTree *t, 
		       const RooArgSet& vars, const RooFormulaVar& cutVar, const char* wgtVarName) :
  RooTreeData(name,title,t,vars,cutVar)
{
  // Constructor of a data set from (part of) an ROOT TTRee. The dimensions
  // of the data set are defined by the 'vars' RooArgSet. For each dimension
  // specified, the TTree must have a branch with the same name. For category
  // branches, this branch should contain the numeric index value. Real dimensions
  // can be constructed from either 'Double_t' or 'Float_t' tree branches. In the
  // latter case, an automatic conversion is applied.
  //
  // The 'cutVar' formula variable
  // is used to select the subset of data points to be copied.
  // For subsets without selection on the data points, or involving cuts
  // operating exclusively and directly on the data set dimensions, the equivalent
  // constructor with a string based cut expression is recommended.

  appendToDir(this,kTRUE) ;

  initialize(wgtVarName) ;
}



//_____________________________________________________________________________
RooDataSet::RooDataSet(const char *name, const char *title, TTree *ntuple, 
		       const RooArgSet& vars, const char *cuts, const char* wgtVarName) :
  RooTreeData(name,title,ntuple,vars,cuts)
{
  // Constructor of a data set from (part of) an ROOT TTRee. The dimensions
  // of the data set are defined by the 'vars' RooArgSet. For each dimension
  // specified, the TTree must have a branch with the same name. For category
  // branches, this branch should contain the numeric index value. Real dimensions
  // can be constructed from either 'Double_t' or 'Float_t' tree branches. In the
  // latter case, an automatic conversion is applied.
  //
  // The 'cuts' string is an optional
  // RooFormula expression and can be used to select the subset of the data points 
  // in 'dset' to be copied. The cut expression can refer to any variable in the
  // vars argset. For cuts involving variables other than those contained in
  // the vars argset, such as intermediate formula objects, use the 
  // equivalent constructor accepting RooFormulaVar reference as cut specification
  //

  appendToDir(this,kTRUE) ;

  initialize(wgtVarName) ;
}



//_____________________________________________________________________________
RooDataSet::RooDataSet(const char *name, const char *filename, const char *treename, 
		       const RooArgSet& vars, const char *cuts, const char* wgtVarName) :
  RooTreeData(name,filename,treename,vars,cuts)
{
  // Constructor of a data set from (part of) a named ROOT TTRee in given ROOT file. 
  // The dimensions of the data set are defined by the 'vars' RooArgSet. For each dimension
  // specified, the TTree must have a branch with the same name. For category
  // branches, this branch should contain the numeric index value. Real dimensions
  // can be constructed from either 'Double_t' or 'Float_t' tree branches. In the
  // latter case, an automatic conversion is applied.
  //
  // The 'cuts' string is an optional
  // RooFormula expression and can be used to select the subset of the data points 
  // in 'dset' to be copied. The cut expression can refer to any variable in the
  // vars argset. For cuts involving variables other than those contained in
  // the vars argset, such as intermediate formula objects, use the 
  // equivalent constructor accepting RooFormulaVar reference as cut specification
  //

  appendToDir(this,kTRUE) ;

  initialize(wgtVarName) ;
}



//_____________________________________________________________________________
RooDataSet::RooDataSet(RooDataSet const & other, const char* newname) :
  RooTreeData(other,newname), RooDirItem()
{
  // Copy constructor

  appendToDir(this,kTRUE) ;
  initialize(other._wgtVar?other._wgtVar->GetName():0) ;
}



//_____________________________________________________________________________
RooArgSet RooDataSet::addWgtVar(const RooArgSet& origVars, const RooAbsArg* wgtVar)
{
  // Helper function for constructor that adds optional weight variable to construct
  // total set of observables

  RooArgSet tmp(origVars) ;
  if (wgtVar) tmp.add(*wgtVar) ;
  return tmp ;
}


//_____________________________________________________________________________
RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
		       const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
		       Int_t nStart, Int_t nStop, Bool_t copyCache) :
  RooTreeData(name,title,ntuple,addWgtVar(vars,ntuple->_wgtVar),cutVar,cutRange,nStart,nStop,copyCache)
{
  // Protected constructor for internal use only

  appendToDir(this,kTRUE) ;
  initialize(ntuple->_wgtVar?ntuple->_wgtVar->GetName():0) ;
}



//_____________________________________________________________________________
RooAbsData* RooDataSet::cacheClone(const RooArgSet* newCacheVars, const char* newName) 
{
  // Return a clone of this dataset containing only the cached variables

  RooDataSet* dset = new RooDataSet(newName?newName:GetName(),GetTitle(),this,_vars,(RooFormulaVar*)0,0,0,2000000000,kTRUE) ;  
  if (_wgtVar) dset->setWeightVar(_wgtVar->GetName()) ;

  RooArgSet* selCacheVars = (RooArgSet*) newCacheVars->selectCommon(dset->_cachedVars) ;
  dset->initCache(*selCacheVars) ;
  delete selCacheVars ;

  return dset ;
}



//_____________________________________________________________________________
RooAbsData* RooDataSet::emptyClone(const char* newName, const char* newTitle, const RooArgSet* vars) const 
{
  // Return an empty clone of this dataset. If vars is not null, only the variables in vars
  // are added to the definition of the empty clone

  // If variables are given, be sure to include weight variable if it exists and is not included
  RooArgSet vars2 ;
  if (vars) {
    vars2.add(*vars) ;
    if (_wgtVar && !vars2.find(_wgtVar->GetName())) {
      vars2.add(*_wgtVar) ;
    }
  } else {
    vars2.add(_vars) ;
  }

  RooDataSet* dset = new RooDataSet(newName?newName:GetName(),newTitle?newTitle:GetTitle(),vars2) ; 
  if (_wgtVar) dset->setWeightVar(_wgtVar->GetName()) ;
  return dset ;
}



//_____________________________________________________________________________
void RooDataSet::initialize(const char* wgtVarName) 
{
  // Initialize the dataset. If wgtVarName is not null, interpret the observable
  // with that name as event weight

  _varsNoWgt.removeAll() ;
  _varsNoWgt.add(_vars) ;
  _wgtVar = 0 ;
  if (wgtVarName) {
    RooAbsArg* wgt = _varsNoWgt.find(wgtVarName) ;
    if (!wgt) {
      coutW(DataHandling) << "RooDataSet::RooDataSet(" << GetName() << ") WARNING: designated weight variable " 
			  << wgtVarName << " not found in set of variables, no weighting will be assigned" << endl ;
    } else if (!dynamic_cast<RooRealVar*>(wgt)) {
      coutW(DataHandling) << "RooDataSet::RooDataSet(" << GetName() << ") WARNING: designated weight variable " 
			  << wgtVarName << " is not of type RooRealVar, no weighting will be assigned" << endl ;
    } else {
      _varsNoWgt.remove(*wgt) ;
      _wgtVar = (RooRealVar*) wgt ;
    }
  }
}



//_____________________________________________________________________________
RooAbsData* RooDataSet::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange, 
				  Int_t nStart, Int_t nStop, Bool_t copyCache)
{
  // Implementation of RooAbsData virtual method that drives the RooAbsData::reduce() methods

  checkInit() ;
  return new RooDataSet(GetName(), GetTitle(), this, varSubset, cutVar, cutRange, nStart, nStop, copyCache) ;

  // WVE - propagate optional weight variable
  //       check behaviour in plotting.

}



//_____________________________________________________________________________
RooDataSet::~RooDataSet()
{
  // Destructor

  removeFromDir(this) ;
}



//_____________________________________________________________________________
RooDataHist* RooDataSet::binnedClone(const char* newName, const char* newTitle) const 
{
  // Return binned clone of this dataset

  TString title, name ;
  if (newName) {
    name = newName ;
  } else {
    name = Form("%s_binned",GetName()) ;
  }
  if (newTitle) {
    title = newTitle ;
  } else {
    title = Form("%s_binned",GetTitle()) ;
  }

  return new RooDataHist(name,title,*get(),*this) ;
}



//_____________________________________________________________________________
void RooDataSet::setWeightVar(const char* name) 
{
  // Designate observable with given name as event weight

  _varsNoWgt.removeAll() ;
  initialize(name) ;
}



//_____________________________________________________________________________
Double_t RooDataSet::weight() const 
{
  // Return event weight of current event

  return _wgtVar ? _wgtVar->getVal() : 1. ;
}



//_____________________________________________________________________________
const RooArgSet* RooDataSet::get(Int_t index) const
{
  // Return RooArgSet with coordinates of event 'index' 
  const RooArgSet* ret  = RooTreeData::get(index) ;
  return ret ? &_varsNoWgt : 0 ;
}



//_____________________________________________________________________________
Int_t RooDataSet::numEntries(Bool_t useWeights) const 
{
  // Return either number of entries (useWeights=false) or
  // rounded sum of weights (useWeights=true). Use the
  // sumEntries() function to get the exact sum of weight

  // Return number of entries if no weights are requested or available
  if (!useWeights || !_wgtVar) return (Int_t) GetEntries() ;

  // Otherwise sum the weights in the event
  return (Int_t)sumEntries() ;
}



//_____________________________________________________________________________
Double_t RooDataSet::sumEntries(const char* cutSpec, const char* cutRange) const 
{
  // Return the sum of weights in all entries matching cutSpec (if specified)
  // and in named range cutRange (if specified)

  // Setup RooFormulaVar for cutSpec if it is present
  RooFormula* select = 0 ;
  if (cutSpec) {
    select = new RooFormula("select",cutSpec,*get()) ;
  }

  // Otherwise sum the weights in the event
  Double_t sumw(0) ;
  Int_t i ;
  for (i=0 ; i<GetEntries() ; i++) {
    get(i) ;
    if (select && select->eval()==0.) continue ;
    if (cutRange && !_vars.allInRange(cutRange)) continue ;
    sumw += weight() ;
  }

  if (select) delete select ;

  return sumw ;  
}



//_____________________________________________________________________________
const RooArgSet* RooDataSet::get() const 
{ 
  // Return a RooArgSet with the coordinates of the current event
  return &_varsNoWgt ; 
} 



//_____________________________________________________________________________
void RooDataSet::add(const RooArgSet& data, Double_t wgt) 
{
  // Add a data point, with its coordinates specified in the 'data' argset, to the data set. 
  // Any variables present in 'data' but not in the dataset will be silently ignored
  //

  checkInit() ;

  _varsNoWgt = data;
  if (_wgtVar) _wgtVar->setVal(wgt) ;
  Fill();
}



//_____________________________________________________________________________
Bool_t RooDataSet::merge(RooDataSet* data1, RooDataSet* data2, RooDataSet* data3, 
			 RooDataSet* data4, RooDataSet* data5, RooDataSet* data6) 
{
  // Merge columns of supplied data set(s) with this data set.  All
  // data sets must have equal number of entries.  In case of
  // duplicate columns the column of the last dataset in the list
  // prevails

  TList dsetList ;
  dsetList.Add(data1) ;
  if (data2) {
    dsetList.Add(data2) ;
    if (data3) {
      dsetList.Add(data3) ;
      if (data4) {
	dsetList.Add(data4) ;
	if (data5) {
	  dsetList.Add(data5) ;
	  if (data6) {
	    dsetList.Add(data6) ;
	  }
	}
      }
    }
  }

  return merge(dsetList) ;
}



//_____________________________________________________________________________
Bool_t RooDataSet::merge(const TList& dsetList) 
{
  // Merge columns of supplied data set(s) with this data set.  All
  // data sets must have equal number of entries.  In case of
  // duplicate columns the column of the last dataset in the list
  // prevails

  checkInit() ;
  
  TIterator* iter = dsetList.MakeIterator() ;
  RooDataSet* data ;

  // Sanity checks: data sets must have the same size
  while((data=(RooDataSet*)iter->Next())) {
    if (numEntries()!=data->numEntries()) {
      coutE(InputArguments) << "RooDataSet::merge(" << GetName() << " ERROR: datasets have different size" << endl ;
      delete iter ;
      return kTRUE ;    
    }
  }

  // Clone current tree
  RooTreeData* cloneData = (RooTreeData*) Clone() ; 

  // Extend vars with elements of other dataset
  iter->Reset() ;
  while((data=(RooDataSet*)iter->Next())) {
    data->_iterator->Reset() ;
    RooAbsArg* arg ;
    while ((arg=(RooAbsArg*)data->_iterator->Next())) {
      RooAbsArg* clone = _vars.addClone(*arg,kTRUE) ;
      if (clone) clone->attachToTree(*_tree,_defTreeBufSize) ;
    }
  }
	   
  // Refill current data set with data of clone and other data set
  Reset() ;
  for (int i=0 ; i<cloneData->numEntries() ; i++) {
    
    // Copy variables from self
    _vars = *cloneData->get(i) ;    

    // Copy variables from merge sets
    iter->Reset() ;
    while((data=(RooDataSet*)iter->Next())) {
      _vars = *data->get(i) ;
    }

    Fill() ;
  }
  
  delete cloneData ;
  delete iter ;

  initialize(_wgtVar?_wgtVar->GetName():0) ;
  return kFALSE ;
}



//_____________________________________________________________________________
void RooDataSet::append(RooTreeData& data) 
{
  // Add all data points of given data set to this data set.
  // Observable in 'data' that are not in this dataset
  // with not be transferred

  checkInit() ;

  loadValues(data._tree,(RooFormulaVar*)0) ;
}



//_____________________________________________________________________________
RooAbsArg* RooDataSet::addColumn(RooAbsArg& var, Bool_t adjustRange) 
{
  // Add a column with the values of the given (function) argument
  // to this dataset. The function value is calculated for each
  // event using the observable values of each event in case the
  // function depends on variables with names that are identical
  // to the observable names in the dataset


  RooAbsArg* ret = RooTreeData::addColumn(var, adjustRange) ;
  initialize(_wgtVar?_wgtVar->GetName():0) ;
  return ret ;
}


//_____________________________________________________________________________
RooArgSet* RooDataSet::addColumns(const RooArgList& varList) 
{
  // Add a column with the values of the given list of (function)
  // argument to this dataset. Each function value is calculated for
  // each event using the observable values of the event in case the
  // function depends on variables with names that are identical to
  // the observable names in the dataset

  RooArgSet* ret = RooTreeData::addColumns(varList) ;
  initialize(_wgtVar?_wgtVar->GetName():0) ;
  return ret ;
}



//_____________________________________________________________________________
TH2F* RooDataSet::createHistogram(const RooAbsRealLValue& var1, const RooAbsRealLValue& var2, const char* cuts, const char *name) const
{
  // Create a TH2F histogram of the distribution of the specified variable
  // using this dataset. Apply any cuts to select which events are used.
  // The variable being plotted can either be contained directly in this
  // dataset, or else be a function of the variables in this dataset.
  // The histogram will be created using RooAbsReal::createHistogram() with
  // the name provided (with our dataset name prepended).

  return createHistogram(var1, var2, var1.getBins(), var2.getBins(), cuts, name);
}



//_____________________________________________________________________________
TH2F* RooDataSet::createHistogram(const RooAbsRealLValue& var1, const RooAbsRealLValue& var2, 
				  Int_t nx, Int_t ny, const char* cuts, const char *name) const
{
  // Create a TH2F histogram of the distribution of the specified variable
  // using this dataset. Apply any cuts to select which events are used.
  // The variable being plotted can either be contained directly in this
  // dataset, or else be a function of the variables in this dataset.
  // The histogram will be created using RooAbsReal::createHistogram() with
  // the name provided (with our dataset name prepended).

  static Int_t counter(0) ;

  Bool_t ownPlotVarX(kFALSE) ;
  // Is this variable in our dataset?
  RooAbsReal* plotVarX= (RooAbsReal*)_vars.find(var1.GetName());
  if(0 == plotVarX) {
    // Is this variable a client of our dataset?
    if (!var1.dependsOn(_vars)) {
      coutE(InputArguments) << GetName() << "::createHistogram: Argument " << var1.GetName() 
			    << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    plotVarX = (RooAbsReal*) var1.Clone()  ;
    ownPlotVarX = kTRUE ;

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    plotVarX->redirectServers(const_cast<RooArgSet&>(_vars)) ;
  }

  Bool_t ownPlotVarY(kFALSE) ;
  // Is this variable in our dataset?
  RooAbsReal* plotVarY= (RooAbsReal*)_vars.find(var2.GetName());
  if(0 == plotVarY) {
    // Is this variable a client of our dataset?
    if (!var2.dependsOn(_vars)) {
      coutE(InputArguments) << GetName() << "::createHistogram: Argument " << var2.GetName() 
			    << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    plotVarY = (RooAbsReal*) var2.Clone()  ;
    ownPlotVarY = kTRUE ;

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    plotVarY->redirectServers(const_cast<RooArgSet&>(_vars)) ;
  }

  // Create selection formula if selection cuts are specified
  RooFormula* select = 0;
  if(0 != cuts && strlen(cuts)) {
    select=new RooFormula(cuts,cuts,_vars);
    if (!select || !select->ok()) {
      delete select;
      return 0 ;
    }
  }
  
  TString histName(name);
  histName.Prepend("_");
  histName.Prepend(fName);
  histName.Append("_") ;
  histName.Append(Form("%08x",counter++)) ;

  // create the histogram
  TH2F* histogram=new TH2F(histName.Data(), "Events", nx, var1.getMin(), var1.getMax(), 
                                                      ny, var2.getMin(), var2.getMax());
  if(!histogram) {
    coutE(DataHandling) << fName << "::createHistogram: unable to create a new histogram" << endl;
    return 0;
  }

  // Dump contents  
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) 
  {
    Int_t entryNumber=_tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);

    if (select && select->eval()==0) continue ;
    histogram->Fill(plotVarX->getVal(), plotVarY->getVal()) ;
  }

  if (ownPlotVarX) delete plotVarX ;
  if (ownPlotVarY) delete plotVarY ;
  if (select) delete select ;

  return histogram ;
}



//_____________________________________________________________________________
RooDataSet *RooDataSet::read(const char *fileList, const RooArgList &varList,
			     const char *verbOpt, const char* commonPath, 
			     const char* indexCatName) {
  // Read given list of ascii files, and construct a data set, using the given
  // ArgList as structure definition.
  //
  // Multiple file names in fileList should be comma separated. Each
  // file is optionally prefixed with 'commonPath' if such a path is
  // provided
  //
  // The arglist specifies the dimensions of the dataset to be built
  // and describes the order in which these dimensions appear in the
  // ascii files to be read. 
  //
  // Each line in the ascii file should contain N white space separated
  // tokens, with N the number of args in 'variables'. Any text beyond
  // N tokens will be ignored with a warning message.
  // [ NB: This format is written by RooArgList::writeToStream() ]
  // 
  // If the value of any of the variables on a given line exceeds the
  // fit range associated with that dimension, the entire line will be
  // ignored. A warning message is printed in each case, unless the
  // 'Q' verbose option is given. (Option 'D' will provide additional
  // debugging information) The number of events read and skipped
  // is always summarized at the end.
  //
  // When multiple files are read, a RooCategory arg in 'variables' can 
  // optionally be designated to hold information about the source file 
  // of each data point. This feature is enabled by giving the name
  // of the (already existing) category variable in 'indexCatName'
  //
  // If no further information is given a label name 'fileNNN' will
  // be assigned to each event, where NNN is the sequential number of
  // the source file in 'fileList'.
  // 
  // Alternatively it is possible to override the default label names
  // of the index category by specifying them in the fileList string:
  // When instead of "file1.txt,file2.txt" the string 
  // "file1.txt:FOO,file2.txt:BAR" is specified, a state named "FOO"
  // is assigned to the index category for each event originating from
  // file1.txt. The labels FOO,BAR may be predefined in the index 
  // category via defineType(), but don't have to be
  //
  // Finally, one can also assign the same label to multiple files,
  // either by specifying "file1.txt:FOO,file2,txt:FOO,file3.txt:BAR"
  // or "file1.txt,file2.txt:FOO,file3.txt:BAR"
  //

  // Make working copy of variables list 
  RooArgList variables(varList) ;

  // Append blinding state category to variable list if not already there
  Bool_t ownIsBlind(kTRUE) ;
  RooAbsArg* blindState = variables.find("blindState") ;
  if (!blindState) {
    blindState = new RooCategory("blindState","Blinding State") ;
    variables.add(*blindState) ;
  } else {
    ownIsBlind = kFALSE ;    
    if (blindState->IsA()!=RooCategory::Class()) {
      oocoutE((TObject*)0,DataHandling) << "RooDataSet::read: ERROR: variable list already contains" 
			  << "a non-RooCategory blindState member" << endl ;
      return 0 ;
    }
    oocoutW((TObject*)0,DataHandling) << "RooDataSet::read: WARNING: recycling existing "
			<< "blindState category in variable list" << endl ;
  }
  RooCategory* blindCat = (RooCategory*) blindState ;

  // Configure blinding state category
  blindCat->setAttribute("Dynamic") ;
  blindCat->defineType("Normal",0) ;
  blindCat->defineType("Blind",1) ;

  // parse the option string
  TString opts= verbOpt;
  opts.ToLower();
  Bool_t verbose= !opts.Contains("q");
  Bool_t debug= opts.Contains("d");
  Bool_t haveRefBlindString(false) ;

  RooDataSet *data= new RooDataSet("dataset", fileList, variables);
  if (ownIsBlind) { variables.remove(*blindState) ; delete blindState ; }
  if(!data) {
    oocoutE((TObject*)0,DataHandling) << "RooDataSet::read: unable to create a new dataset"
			<< endl;
    return 0;
  }

  // Redirect blindCat to point to the copy stored in the data set
  blindCat = (RooCategory*) data->_vars.find("blindState") ;

  // Find index category, if requested
  RooCategory *indexCat     = 0;
  //RooCategory *indexCatOrig = 0;
  if (indexCatName) { 
    RooAbsArg* tmp = 0;
    tmp = data->_vars.find(indexCatName) ;
    if (!tmp) {
      oocoutE((TObject*)0,DataHandling) << "RooDataSet::read: no index category named " 
			  << indexCatName << " in supplied variable list" << endl ;
      return 0 ;
    }
    if (tmp->IsA()!=RooCategory::Class()) {
      oocoutE((TObject*)0,DataHandling) << "RooDataSet::read: variable " << indexCatName 
			  << " is not a RooCategory" << endl ;
      return 0 ;
    }
    indexCat = (RooCategory*)tmp ;
    
    // Prevent RooArgSet from attempting to read in indexCat
    indexCat->setAttribute("Dynamic") ;
  }


  Int_t outOfRange(0) ;

  // Make local copy of file list for tokenizing 
  char fileList2[10240] ;
  strcpy(fileList2,fileList) ;
  
  // Loop over all names in comma separated list
  char *filename = strtok(fileList2,", ") ;
  Int_t fileSeqNum(0) ;
  while (filename) {
    // Determine index category number, if this option is active
    if (indexCat) {

      // Find and detach optional file category name 
      char *catname = strchr(filename,':') ;

      if (catname) {
	// Use user category name if provided
	*catname=0 ;
	catname++ ;

	const RooCatType* type = indexCat->lookupType(catname,kFALSE) ;
	if (type) {
	  // Use existing category index
	  indexCat->setIndex(type->getVal()) ;
	} else {
	  // Register cat name
	  indexCat->defineType(catname,fileSeqNum) ;
	  indexCat->setIndex(fileSeqNum) ;
	}
      } else {
	// Assign autogenerated name
	char newLabel[128] ;
	sprintf(newLabel,"file%03d",fileSeqNum) ;
	if (indexCat->defineType(newLabel,fileSeqNum)) {
	  oocoutE((TObject*)0,DataHandling) << "RooDataSet::read: Error, cannot register automatic type name " << newLabel 
			      << " in index category " << indexCat->GetName() << endl ;
	  return 0 ;
	}	
	// Assign new category number
	indexCat->setIndex(fileSeqNum) ;
      }
    }

    oocoutI((TObject*)0,DataHandling) << "RooDataSet::read: reading file " << filename << endl ;

    // Prefix common path 
    TString fullName(commonPath) ;
    fullName.Append(filename) ;
    ifstream file(fullName) ;

    if(!file.good()) {
      oocoutW((TObject*)0,DataHandling) << "RooDataSet::read: unable to open '"
	   << filename << "', skipping" << endl;
    }
    
//  Double_t value;
    Int_t line(0) ;
    Bool_t haveBlindString(false) ;

    while(file.good() && !file.eof()) {
      line++;
      if(debug) oocxcoutD((TObject*)0,DataHandling) << "reading line " << line << endl;

      // process comment lines
      if (file.peek() == '#')
	{
	  if(debug) oocxcoutD((TObject*)0,DataHandling) << "skipping comment on line " << line << endl;
	    
	  TString aline ;
	  aline.ReadLine(file) ;
	  if (aline.Contains("#BLIND#")) {	  
	    haveBlindString = true ;
	    if (haveRefBlindString) {
	      
	      // compare to ref blind string 
	      TString curBlindString(aline(7,aline.Length()-7)) ;
	      if (debug) oocxcoutD((TObject*)0,DataHandling) << "Found blind string " << curBlindString << endl ;
	      if (curBlindString != data->_blindString) {
		  oocoutE((TObject*)0,DataHandling) << "RooDataSet::read: ERROR blinding string mismatch, abort" << endl ;
		  return 0 ;
		}
	      
	    } else {
	      // store ref blind string 
	      data->_blindString=TString(aline(7,aline.Length()-7)) ;
	      if (debug) oocxcoutD((TObject*)0,DataHandling) << "Storing ref blind string " << data->_blindString << endl ;
	      haveRefBlindString = true ;
	    }	    
	  }     
	}
      else {	

	// Skip empty lines 
	// if(file.peek() == '\n') { file.get(); }

	// Read single line
	Bool_t readError = variables.readFromStream(file,kTRUE,verbose) ;
	data->_vars = variables ;
// 	Bool_t readError = data->_vars.readFromStream(file,kTRUE,verbose) ;

	// Stop at end of file or on read error
	if(file.eof()) break ;	
	if(!file.good()) {
	  oocoutE((TObject*)0,DataHandling) << "RooDataSet::read(static): read error at line " << line << endl ;
	  break;
	}	

	if (readError) {
	  outOfRange++ ;
	  continue ;
	}
	blindCat->setIndex(haveBlindString) ;
	data->Fill(); // store this event
      }
    }

    file.close();

    // get next file name 
    filename = strtok(0," ,") ;
    fileSeqNum++ ;
  }

  if (indexCat) {
    // Copy dynamically defined types from new data set to indexCat in original list
    RooCategory* origIndexCat = (RooCategory*) variables.find(indexCatName) ;
    TIterator* tIter = indexCat->typeIterator() ;
    RooCatType* type = 0;
      while ((type=(RooCatType*)tIter->Next())) {
	origIndexCat->defineType(type->GetName(),type->getVal()) ;
      }
  }
  oocoutI((TObject*)0,DataHandling) << "RooDataSet::read: read " << data->GetEntries()
				    << " events (ignored " << outOfRange << " out of range events)" << endl;
  return data;
}




//_____________________________________________________________________________
Bool_t RooDataSet::write(const char* filename)
{
  // Write the contents of this dataset to an ASCII file with the specified name
  // Each event will be written as a single line containing the written values
  // of each observable in the order they were declared in the dataset and
  // separated by whitespaces

  checkInit() ;

  // Open file for writing 
  ofstream ofs(filename) ;
  if (ofs.fail()) {
    coutE(DataHandling) << "RooDataSet::write(" << GetName() << ") cannot create file " << filename << endl ;
    return kTRUE ;
  }

  // Write all lines as arglist in compact mode
  coutI(DataHandling) << "RooDataSet::write(" << GetName() << ") writing ASCII file " << filename << endl ;
  Int_t i ;
  for (i=0 ; i<numEntries() ; i++) {
    RooArgList list(*get(i),"line") ;
    list.writeToStream(ofs,kTRUE) ;
  }

  if (ofs.fail()) {
    coutW(DataHandling) << "RooDataSet::write(" << GetName() << "): WARNING error(s) have occured in writing" << endl ;
  }
  return ofs.fail() ;
}



//_____________________________________________________________________________
void RooDataSet::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const 
{
  // Print info about this dataset to the specified output stream.
  //
  //   Standard: number of entries
  //      Shape: list of variables we define & were generated with

  RooTreeData::printMultiline(os,contents,verbose,indent) ;
  if (_wgtVar) {
    os << indent << "  Dataset variable \"" << _wgtVar->GetName() << "\" is interpreted as the event weight" << endl ;
  }
}


//_____________________________________________________________________________
void RooDataSet::printValue(ostream& os) const 
{
  // Print value of the dataset, i.e. the sum of weights contained in the dataset
  os << numEntries(kFALSE) << " entries" ;
  if (isWeighted()) {
    os << " (" << sumEntries() << " weighted)" ;
  }
}



//_____________________________________________________________________________
void RooDataSet::printArgs(ostream& os) const 
{
  // Print argument of dataset, i.e. the observable names

  os << "[" ;    
  TIterator* iter = _varsNoWgt.createIterator() ;
  RooAbsArg* arg ;
  Bool_t first(kTRUE) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (first) {
      first=kFALSE ;
    } else {
      os << "," ;
    }
    os << arg->GetName() ;
  }
  if (_wgtVar) {
    os << ",weight:" << _wgtVar->GetName() ;
  }
  os << "]" ;
  delete iter ;
}



//_____________________________________________________________________________
void RooDataSet::SetName(const char *name) 
{
  // Change the name of this dataset into the given name

  if (_dir) _dir->GetList()->Remove(this);
  TNamed::SetName(name) ;
  if (_dir) _dir->GetList()->Add(this);
}


//_____________________________________________________________________________
void RooDataSet::SetNameTitle(const char *name, const char* title) 
{
  // Change the title of this dataset into the given name

  if (_dir) _dir->GetList()->Remove(this);
  TNamed::SetNameTitle(name,title) ;
  if (_dir) _dir->GetList()->Add(this);
}
