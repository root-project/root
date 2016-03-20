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

/**
\file RooFormula.cxx
\class RooFormula
\ingroup Roofitcore

RooFormula an implementation of ROOT::v5::TFormula that interfaces it to RooAbsArg
value objects. It allows to use the value of a given list of RooAbsArg objects in the formula
expression. Reference is done either by the RooAbsArgs name
or by list ordinal postion ('@0,@1,...'). State information
of RooAbsCategories can be accessed used the '::' operator,
e.g. 'tagCat::Kaon' will resolve to the numerical value of
the 'Kaon' state of the RooAbsCategory object named tagCat.
**/

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include "TROOT.h"
#include "TClass.h"
#include "TObjString.h"
#include "RooFormula.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "RooTrace.h"

using namespace std;

ClassImp(RooFormula)


////////////////////////////////////////////////////////////////////////////////
/// Default constructor
/// coverity[UNINIT_CTOR]

RooFormula::RooFormula() : ROOT::v5::TFormula(), _nset(0)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with expression string and list of RooAbsArg variables

RooFormula::RooFormula(const char* name, const char* formula, const RooArgList& list) : 
  ROOT::v5::TFormula(), _isOK(kTRUE), _compiled(kFALSE)
{
  SetName(name) ;
  SetTitle(formula) ;

  TIterator* iter = list.createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    _origList.Add(arg) ;
  }
  delete iter ;

  _compiled = kTRUE ;
  if (Compile()) {
    coutE(InputArguments) << "RooFormula::RooFormula(" << GetName() << "): compile error" << endl ;
    _isOK = kFALSE ;
    return ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooFormula::RooFormula(const RooFormula& other, const char* name) : 
  ROOT::v5::TFormula(), RooPrintable(other), _isOK(other._isOK), _compiled(kFALSE) 
{
  SetName(name?name:other.GetName()) ;
  SetTitle(other.GetTitle()) ;

  TIterator* iter = other._origList.MakeIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    _origList.Add(arg) ;
  }
  delete iter ;
  
  Compile() ;
  _compiled=kTRUE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Recompile formula with new expression

Bool_t RooFormula::reCompile(const char* newFormula) 
{
  fNval=0 ;
  _useList.Clear() ;  

  TString oldFormula=GetTitle() ;
  if (Compile(newFormula)) {
    coutE(InputArguments) << "RooFormula::reCompile: new equation doesn't compile, formula unchanged" << endl ;
    reCompile(oldFormula) ;    
    return kTRUE ;
  }

  SetTitle(newFormula) ;
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooFormula::~RooFormula() 
{
  _labelList.Delete() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return list of RooAbsArg dependents that is actually used by formula expression

RooArgSet& RooFormula::actualDependents() const
{
  if (!_compiled) {
    _isOK = !((RooFormula*)this)->Compile() ;
    _compiled = kTRUE ;
  }

  // Return list of dependents used in formula expression

  _actual.removeAll();
  
  int i ;
  for (i=0 ; i<_useList.GetSize() ; i++) {
    _actual.add((RooAbsArg&)*_useList.At(i),kTRUE) ;
  }

  return _actual ;
}



////////////////////////////////////////////////////////////////////////////////
/// DEBUG: Dump state information

void RooFormula::dump() 
{
  int i ;
  cout << "RooFormula::dump()" << endl ;
  cout << "useList:" << endl ;
  for (i=0 ; i<_useList.GetSize() ; i++) {
    cout << "[" << i << "] = " << (void*) _useList.At(i) << " " << _useList.At(i)->GetName() << endl ;
  }
  cout << "labelList:" << endl ;
  for (i=0 ; i<_labelList.GetSize() ; i++) {
    cout << "[" << i << "] = " << (void*) _labelList.At(i) << " " << _labelList.At(i)->GetName() <<  endl ;
  }
  cout << "origList:" << endl ;
  for (i=0 ; i<_origList.GetSize() ; i++) {
    cout << "[" << i << "] = " << (void*) _origList.At(i)  << " " << _origList.At(i)->GetName() <<  endl ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Change used variables to those with the same name in given list
/// If mustReplaceAll is true and error is generated if one of the
/// elements of newDeps is not found as a server

Bool_t RooFormula::changeDependents(const RooAbsCollection& newDeps, Bool_t mustReplaceAll, Bool_t nameChange) 
{
  //Change current servers to new servers with the same name given in list
  Bool_t errorStat(kFALSE) ;
  int i ;

  for (i=0 ; i<_useList.GetSize() ; i++) {
    RooAbsReal* replace = (RooAbsReal*) ((RooAbsArg*)_useList.At(i))->findNewServer(newDeps,nameChange) ;
    if (replace) {
      _useList.Replace(_useList.At(i),replace) ;
    } else if (mustReplaceAll) {
      coutE(LinkStateMgmt) << "RooFormula::changeDependents(1): cannot find replacement for " 
			   << _useList.At(i)->GetName() << endl ;
      errorStat = kTRUE ;
    }
  }  

  TIterator* iter = _origList.MakeIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    RooAbsReal* replace = (RooAbsReal*) arg->findNewServer(newDeps,nameChange) ;
    if (replace) {
      _origList.Replace(arg,replace) ;
      if (arg->getStringAttribute("origName")) {
	replace->setStringAttribute("origName",arg->getStringAttribute("origName")) ;
      } else {
	replace->setStringAttribute("origName",arg->GetName()) ;
      }
    } else if (mustReplaceAll) {
      errorStat = kTRUE ;
    }
  }
  delete iter ;

  return errorStat ;
}



////////////////////////////////////////////////////////////////////////////////
/// Evaluate ROOT::v5::TFormula using given normalization set to be used as
/// observables definition passed to RooAbsReal::getVal()

Double_t RooFormula::eval(const RooArgSet* nset)
{ 
  if (!_compiled) {
    _isOK = !Compile() ;
    _compiled = kTRUE ;
  }

  // WVE sanity check should go here
  if (!_isOK) {
    coutE(Eval) << "RooFormula::eval(" << GetName() << "): Formula doesn't compile: " << GetTitle() << endl ;
    return 0. ;
  }

  // Pass current dataset pointer to DefinedValue
  _nset = (RooArgSet*) nset ;

  return EvalPar(0,0) ; 
}


Double_t

////////////////////////////////////////////////////////////////////////////////
/// Interface to ROOT::v5::TFormula, return value defined by object with id 'code'
/// Object ids are mapped from object names by method DefinedVariable()

RooFormula::DefinedValue(Int_t code) 
{
  // Return current value for variable indicated by internal reference code
  if (code>=_useList.GetSize()) return 0 ;

  RooAbsArg* arg=(RooAbsArg*)_useList.At(code) ;
  if (_useIsCat[code]) {

    // Process as category
    const RooAbsCategory *absCat = (const RooAbsCategory*)(arg);
    TString& label=((TObjString*)_labelList.At(code))->String() ;
    if (label.IsNull()) {
      return absCat->getIndex() ;
    } else {
      return absCat->lookupType(label)->getVal() ; // DK: why not call getVal(_nset) here also?
    }

  } else {

    // Process as real 
    const RooAbsReal *absReal= (const RooAbsReal*)(arg);  
    return absReal->getVal(_nset) ;
    
  }

  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface to ROOT::v5::TFormula. If name passed by ROOT::v5::TFormula is recognized
/// as one of our RooAbsArg servers, return a unique id integer
/// that represent this variable.

Int_t RooFormula::DefinedVariable(TString &name, int& action)
{
  Int_t ret = DefinedVariable(name) ;
  if (ret>=0) {

#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
     action = kDefinedVariable;
#else
     action = 0 ; // prevents compiler warning
#endif

  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface to ROOT::v5::TFormula. If name passed by ROOT::v5::TFormula is recognized
/// as one of our RooAbsArg servers, return a unique id integer
/// that represent this variable.

Int_t RooFormula::DefinedVariable(TString &name) 
{
  char argName[1024];
  strlcpy(argName,name.Data(),1024) ;

  // Find :: operator and split string if found
  char *labelName = strstr(argName,"::") ;
  if (labelName) {
    *labelName = 0 ;
    labelName+= 2 ;
  }

  // Defined internal reference code for given named variable 
  RooAbsArg *arg = 0;
  if (argName[0] == '@') {
    // Access by ordinal number
    Int_t index = atoi(argName+1) ;
    if (index>=0 && index<_origList.GetSize()) {
      arg = (RooAbsArg*) _origList.At(index) ;
    } else {
      coutE(Eval) << "RooFormula::DefinedVariable(" << GetName() 
		  << ") ERROR: ordinal variable reference " << name 
		  << " out of range (0 - " << _origList.GetSize()-1 << ")" << endl ;
    }
  } else {
    // Access by name
    arg= (RooAbsArg*) _origList.FindObject(argName) ;
    if (!arg) {
      for (RooLinkedListIter it = _origList.iterator(); RooAbsArg* v = dynamic_cast<RooAbsArg*>(it.Next());) {
	if (!TString(argName).CompareTo(v->getStringAttribute("origName"))) {
	  arg= v ;
	}
      }
    }
  }

  // Check that arg exists
  if (!arg) return -1 ;

  // Check that optional label corresponds to actual category state
  if (labelName) {
    RooAbsCategory* cat = dynamic_cast<RooAbsCategory*>(arg) ;
    if (!cat) {
      coutE(Eval) << "RooFormula::DefinedVariable(" << GetName() << ") ERROR: " 
		  << arg->GetName() << "' is not a RooAbsCategory" << endl ;
      return -1 ;
    }

    if (!cat->lookupType(labelName)) {
      coutE(Eval) << "RooFormula::DefinedVariable(" << GetName() << ") ERROR '" 
		  << labelName << "' is not a state of " << arg->GetName() << endl ;
      return -1 ;
    }

  }


  // Check if already registered
  Int_t i ;
  for(i=0 ; i<_useList.GetSize() ; i++) {
    RooAbsArg* var = (RooAbsArg*) _useList.At(i) ;
    //Bool_t varMatch = !TString(var->GetName()).CompareTo(arg->GetName()) ;
    Bool_t varMatch = !TString(var->GetName()).CompareTo(arg->GetName()) && !TString(var->getStringAttribute("origName")).CompareTo(arg->GetName());

    if (varMatch) {
      TString& lbl= ((TObjString*) _labelList.At(i))->String() ;
      Bool_t lblMatch(kFALSE) ;
      if (!labelName && lbl.IsNull()) {
	lblMatch=kTRUE ;
      } else if (labelName && !lbl.CompareTo(labelName)) {
	lblMatch=kTRUE ;
      }

      if (lblMatch) {
	// Label and variable name match, recycle entry
	return i ;
      }
    }
  }

  // Register new entry ;
  _useList.Add(arg) ;
  _useIsCat.push_back(dynamic_cast<RooAbsCategory*>(arg)) ;
  if (!labelName) {
    _labelList.Add(new TObjString("")) ;
  } else {
    _labelList.Add(new TObjString(labelName)) ;
  }

   return (_useList.GetSize()-1) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Printing interface

void RooFormula::printMultiline(ostream& os, Int_t /*contents*/, Bool_t /*verbose*/, TString indent) const 
{
  os << indent << "--- RooFormula ---" << endl;
  os << indent << "  Formula: \"" << GetTitle() << "\"" << endl;
  indent.Append("  ");
  os << indent << actualDependents() << endl ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print value of formula

void RooFormula::printValue(ostream& os) const 
{
  os << const_cast<RooFormula*>(this)->eval(0) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print name of formula

void RooFormula::printName(ostream& os) const 
{
  os << GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print title of formula

void RooFormula::printTitle(ostream& os) const 
{
  os << GetTitle() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print class name of formula

void RooFormula::printClassName(ostream& os) const 
{
  os << IsA()->GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print arguments of formula, i.e. dependents that are actually used

void RooFormula::printArgs(ostream& os) const 
{
  os << "[ actualVars=" << _actual << " ]" ;
}
