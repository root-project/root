/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooFormula.cc,v 1.13 2001/05/03 02:15:55 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, University of California Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include "TROOT.h"
#include "TClass.h"
#include "TObjString.h"
#include "RooFitCore/RooFormula.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooAbsCategory.hh"

ClassImp(RooFormula)

RooFormula::RooFormula() : TFormula()
{
}

RooFormula::RooFormula(const char* name, const char* formula, const RooArgSet& list) : 
  TFormula(), _isOK(kTRUE) 
{
  SetName(name) ;
  SetTitle(formula) ;

  TIterator* iter = list.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _origList.Add(arg) ;
  }

  if (Compile()) {
    cout << "RooFormula::RooFormula(" << GetName() << "): compile error" << endl ;
    _isOK = kFALSE ;
    return ;
  }
}


RooFormula::RooFormula(const RooFormula& other, const char* name) : 
  TFormula(), _isOK(other._isOK) 
{
  SetName(name?name:other.GetName()) ;
  SetTitle(other.GetTitle()) ;

  TIterator* iter = other._origList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _origList.Add(arg) ;
  }
  
  Compile() ;
}



Bool_t RooFormula::reCompile(const char* newFormula) 
{
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
  // to be implemented
  return *this ;
}


RooFormula::~RooFormula() 
{
}



RooArgSet& RooFormula::actualDependents() const
{
  // Return list of actual dependents
  // (formula might not use all given input parameters)
  static RooArgSet set("actualDependents") ;
  set.Clear() ;
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


Bool_t RooFormula::changeDependents(const RooArgSet& newDeps, Bool_t mustReplaceAll) 
{
  //Change current servers to new servers with the same name given in list
  Bool_t errorStat(kFALSE) ;
  int i ;
  for (i=0 ; i<_useList.GetEntries() ; i++) {
    RooAbsReal* replace = (RooAbsReal*) newDeps.find(_useList[i]->GetName()) ;
    if (replace) {
      _useList[i] = replace ;
    } else if (mustReplaceAll) {
      cout << "RooFormula::changeDependents: cannot find replacement for " 
	   << _useList[i]->GetName() << endl ;
      errorStat = kTRUE ;
    }
  }  
  return errorStat ;
}


Double_t RooFormula::eval()
{ 
  // WVE sanity check should go here
  if (!_isOK) {
    cout << "RooFormula::eval(" << GetName() << "): Formula doesn't compile" << endl ;
    return 0. ;
  }
  return EvalPar(0,0) ; 
}


Double_t
RooFormula::DefinedValue(Int_t code) {
  // Return current value for parameter indicated by internal reference code
  if (code>=_useList.GetEntries()) return 0 ;
  RooAbsArg* arg=(RooAbsArg*)_useList.At(code) ;
  TString& label=((TObjString*)_labelList.At(code))->String() ;

  if (arg->IsA()->InheritsFrom(RooAbsReal::Class())) {
    return ((RooAbsReal*)arg)->getVal() ;
  } else if (arg->IsA()->InheritsFrom(RooAbsCategory::Class())) {
    if (label.IsNull()) {
      return ((RooAbsCategory*)_useList.At(code))->getIndex() ;
    } else {
      return ((RooAbsCategory*)_useList.At(code))->lookupType(label)->getVal() ;
    }
  }
}


Int_t 
RooFormula::DefinedVariable(TString &name)
{
  char argName[1024];
  strcpy(argName,name.Data()) ;

  // Find :: operator and split string if found
  char *labelName = strstr(argName,"::") ;
  if (labelName) {
    *labelName = 0 ;
    labelName+= 2 ;
  }

  // Defined internal reference code for given named variable 
  RooAbsArg* arg= (RooAbsArg*) _origList.FindObject(argName) ;
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
