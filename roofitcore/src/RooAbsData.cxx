/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsData.cc,v 1.11 2001/10/19 06:56:51 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   15-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [DATA] --
// RooAbsData is the common abstract base class for binned and unbinned
// datasets. The abstract interface defines plotting and tabulating entry
// points for its contents and provides an iterator over its elements
// (bins for binned data sets, data points for unbinned datasets).

#include "RooFitCore/RooAbsData.hh"

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
      cout << "RooDataSet::initialize(" << GetName() 
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
  TNamed(newname?newname:other.GetName(),other.GetTitle()), _vars(other._vars),
  _cachedVars("Cached Variables"), _doDirtyProp(kTRUE)
{
  // Copy constructor
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



RooAbsData* RooAbsData::reduce(const char* cut) 
{ 
  // Create a subset of the data set by applying the given cut on the data points.
  // The cut expression can refer to any variable in the data set. For cuts involving 
  // other variables, such as intermediate formula objects, use the equivalent 
  // reduce method specifying the as a RooFormulVar reference.

  RooFormulaVar cutVar(cut,cut,*get()) ;
  return reduceEng(*get(),&cutVar,kFALSE) ;
}




RooAbsData* RooAbsData::reduce(const RooFormulaVar& cutVar) 
{
  // Create a subset of the data set by applying the given cut on the data points.
  // The 'cutVar' formula variable is used to select the subset of data points to be 
  // retained in the reduced data collection.
  return reduceEng(*get(),&cutVar,kFALSE) ;
}




RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const char* cut) 
{
  // Create a subset of the data set by applying the given cut on the data points
  // and reducing the dimensions to the specified set.
  // 
  // The cut expression can refer to any variable in the data set. For cuts involving 
  // other variables, such as intermediate formula objects, use the equivalent 
  // reduce method specifying the as a RooFormulVar reference.

  if (cut) {
    RooFormulaVar cutVar(cut,cut,*get()) ;
    return reduceEng(varSubset,&cutVar,kFALSE) ;      
  } 
  return reduceEng(varSubset,0,kFALSE) ;
}




RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const RooFormulaVar& cutVar) 
{
  // Create a subset of the data set by applying the given cut on the data points
  // and reducing the dimensions to the specified set.
  // 
  // The 'cutVar' formula variable is used to select the subset of data points to be 
  // retained in the reduced data collection.
  return reduceEng(varSubset,&cutVar,kFALSE) ;
}

