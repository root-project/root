/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooFormula.cc,v 1.26 2001/09/11 00:30:32 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, University of California Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooFormula is the RFC extension of TFormula. It allows to use
// the value of a given list of RooAbsArg objects in the formula
// expression. Reference is done either by the RooAbsArgs name
// or by list ordinal postion ('@0,@1,...'). State information
// of RooAbsCategories can be accessed used the '::' operator,
// e.g. 'tagCat::Kaon' will resolve to the numerical value of
// the Kaon state of the RooAbsCategory object named tagCat.

#include <iostream.h>
#include <stdlib.h>
#include "TROOT.h"
#include "TClass.h"
#include "TObjString.h"
#include "RooFitCore/RooFormula.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooAbsCategory.hh"

ClassImp(RooFormula)

RooFormula::RooFormula() : TFormula(), _nset(0)
{
  // Dummy constructor
}

RooFormula::RooFormula(const char* name, const char* formula, const RooArgSet& list) : 
  TFormula(), _isOK(kTRUE)
{
  // Constructor with expression string and list of variables
  SetName(name) ;
  SetTitle(formula) ;

  TIterator* iter = list.createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _origList.Add(arg) ;
  }
  delete iter ;

  if (Compile()) {
    cout << "RooFormula::RooFormula(" << GetName() << "): compile error" << endl ;
    _isOK = kFALSE ;
    return ;
  }
}


RooFormula::RooFormula(const RooFormula& other, const char* name) : 
  TFormula(), _isOK(other._isOK)
{
  // Copy constructor

  SetName(name?name:other.GetName()) ;
  SetTitle(other.GetTitle()) ;

  TIterator* iter = other._origList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _origList.Add(arg) ;
  }
  delete iter ;
  
  Compile() ;
}



Bool_t RooFormula::reCompile(const char* newFormula) 
{
  // Recompile formula with new expression

  fNval=0 ;
  _useList.Clear() ;  

  TString oldFormula=GetTitle() ;
  if (Compile(newFormula)) {
    cout << "RooFormula::reCompile: new equation doesn't compile, formula unchanged" << endl ;
    reCompile(oldFormula) ;    
    return kTRUE ;
  }

  SetTitle(newFormula) ;
  return kFALSE ;
}


RooFormula& RooFormula::operator=(const RooFormula& other) 
{
  // to be deprecated

  return *this ;
}


RooFormula::~RooFormula() 
{
  // Destructor
  _labelList.Delete() ;
}



RooArgSet& RooFormula::actualDependents() const
{
  // Return list of dependents used in formula expression

  // (formula might not use all given input parameters)
  static RooArgSet set("actualDependents") ;
  set.removeAll();
  int i ;
  for (i=0 ; i<_useList.GetEntries() ; i++) {
    set.add((RooAbsArg&)*_useList.At(i),kTRUE) ;
  }
  return set ;
}


void RooFormula::dump() {
  // DEBUG: Dump state information
  int i ;
  cout << "RooFormula::dump()" << endl ;
  cout << "useList:" << endl ;
  for (i=0 ; i<_useList.GetEntries() ; i++) {
    cout << "[" << i << "] = " << (void*) _useList.At(i) << " " << _useList.At(i)->GetName() << endl ;
  }
  cout << "labelList:" << endl ;
  for (i=0 ; i<_labelList.GetEntries() ; i++) {
    cout << "[" << i << "] = " << (void*) _labelList.At(i) << _useList.At(i)->GetName() <<  endl ;
  }
  cout << "origList:" << endl ;
  for (i=0 ; i<_origList.GetSize() ; i++) {
    cout << "[" << i << "] = " << (void*) _origList.At(i)  << _useList.At(i)->GetName() <<  endl ;
  }
}


Bool_t RooFormula::changeDependents(const RooAbsCollection& newDeps, Bool_t mustReplaceAll) 
{
  // Change used variables to those with the same name in given list

  //Change current servers to new servers with the same name given in list
  Bool_t errorStat(kFALSE) ;
  int i ;

  for (i=0 ; i<_useList.GetEntries() ; i++) {
    RooAbsReal* replace = (RooAbsReal*) newDeps.find(_useList[i]->GetName()) ;
    if (replace) {
      _useList[i] = replace ;
    } else if (mustReplaceAll) {
      cout << "RooFormula::changeDependents(1): cannot find replacement for " 
	   << _useList[i]->GetName() << endl ;
      errorStat = kTRUE ;
    }
  }  

  TIterator* iter = _origList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    RooAbsReal* replace = (RooAbsReal*) newDeps.find(arg->GetName()) ;
    if (replace) {
      _origList.AddBefore(arg,replace) ;
      _origList.Remove(arg) ;
    } else if (mustReplaceAll) {
//       cout << "RooFormula::changeDependents(3): cannot find replacement for " 
// 	   << arg->GetName() << "(" << arg << ")" << endl ;
      errorStat = kTRUE ;
    }
  }
  delete iter ;

  return errorStat ;
}


Double_t RooFormula::eval(const RooArgSet* nset)
{ 
  // Return current value of formula  

  // WVE sanity check should go here
  if (!_isOK) {
    cout << "RooFormula::eval(" << GetName() << "): Formula doesn't compile" << endl ;
    return 0. ;
  }

  // Pass current dataset pointer to DefinedValue
  _nset = (RooArgSet*) nset ;

  return EvalPar(0,0) ; 
}


Double_t
RooFormula::DefinedValue(Int_t code) {
  // Return current value for variable indicated by internal reference code
  if (code>=_useList.GetEntries()) return 0 ;
  RooAbsArg* arg=(RooAbsArg*)_useList.At(code) ;

  const RooAbsReal *absReal= dynamic_cast<const RooAbsReal*>(arg);
  if(0 != absReal) {
    return absReal->getVal(_nset) ;
  } else {
    const RooAbsCategory *absCat= dynamic_cast<const RooAbsCategory*>(arg);
    if(0 != absCat) {
      TString& label=((TObjString*)_labelList.At(code))->String() ;
      if (label.IsNull()) {
	return absCat->getIndex() ;
      } else {
	return absCat->lookupType(label)->getVal() ; // DK: why not call getVal(_nset) here also?
      }
    }
  }
  assert(0) ;
  return 0 ;
}


Int_t 
RooFormula::DefinedVariable(TString &name)
{
  // Check if a variable with given name is available

  char argName[1024];
  strcpy(argName,name.Data()) ;

  // Find :: operator and split string if found
  char *labelName = strstr(argName,"::") ;
  if (labelName) {
    *labelName = 0 ;
    labelName+= 2 ;
  }

  // Defined internal reference code for given named variable 
  RooAbsArg *arg(0) ;
  if (argName[0] == '@') {
    // Access by ordinal number
    Int_t index = atoi(argName+1) ;
    if (index>=0 && index<_origList.GetSize()) {
      arg = (RooAbsArg*) _origList.At(index) ;
    } else {
      cout << "RooFormula::DefinedVariable(" << GetName() 
	   << ") ERROR: ordinal variable reference " << name 
	   << " out of range (0 - " << _origList.GetSize()-1 << ")" << endl ;
    }
  } else {
    // Access by name
    arg= (RooAbsArg*) _origList.FindObject(argName) ;
  }

  if (!arg) return -1 ;

  // Add variable to use list
  _useList.Add(arg) ;
  if (labelName) {
    _labelList.Add(new TObjString(labelName)) ;
  } else {
    _labelList.Add(new TObjString("")) ;
  }

  return (_useList.GetEntries()-1) ;
}


void RooFormula::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this argument set to the specified stream.
  //
  //   OneLine: use RooPrintable::oneLinePrint()
  //  Standard: our formula
  //   Verbose: formula and list of actual dependents

  if(opt == Standard) {
    os << indent << GetTitle() << endl;
  }
  else {
    oneLinePrint(os,*this);
    if(opt == Verbose) {
      os << indent << "--- RooFormula ---" << endl;
      os << indent << "  Formula: \"" << GetTitle() << "\"" << endl;
      indent.Append("  ");
      os << indent;
      actualDependents().printToStream(os,lessVerbose(opt),indent);
    }
  }
}
