/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsData.cc,v 1.4 2001/08/23 01:35:14 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   15-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooAbsData is the common abstract base class for binned and unbinned
// datasets. The abstract interface defines plotting and tabulating entry
// points for its contents and provides an iterator over its elements
// (bins for binned data sets, ttree rows for unbinned datasets).

#include "RooFitCore/RooAbsData.hh"

ClassImp(RooAbsData)
;


RooAbsData::RooAbsData() 
{
  // Default constructor
}


RooAbsData::RooAbsData(const char *name, const char *title, const RooArgSet& vars) :
  TNamed(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), 
  _doDirtyProp(kTRUE)

{
  // Constructor from a list of variables. Only fundamental elements of vars
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
      RooAbsArg* varClone = (RooAbsArg*) var->Clone() ;
      _vars.add(*varClone) ;
    }
  }
  delete iter ;

  _iterator= _vars.createIterator();
  _cacheIter = _cachedVars.createIterator() ;
}




RooAbsData::RooAbsData(const RooAbsData& other, const char* newname) : 
  TNamed(newname?newname:GetName(),other.GetTitle()), _vars(other._vars)
{
  // Copy constructor
}




RooAbsData::~RooAbsData() 
{
  // Destructor, delete owned contents.
  delete _iterator ;
  delete _cacheIter ;
}


