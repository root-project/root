/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   18-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

#include "RooFitCore/RooCmdConfig.hh"
#include "RooFitCore/RooInt.hh"
#include "RooFitCore/RooDouble.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooTObjWrap.hh"
#include "RooFitCore/RooAbsData.hh"
#include "TObjString.h"

ClassImp(RooCmdConfig) 
  ;


RooCmdConfig::RooCmdConfig(const char* methodName) :
  _name(methodName)
{
  _verbose = kFALSE ;
  _error = kFALSE ;
  _allowUndefined = kFALSE ;

  _iIter = _iList.MakeIterator() ;
  _dIter = _dList.MakeIterator() ;
  _sIter = _sList.MakeIterator() ;
  _oIter = _oList.MakeIterator() ;

  _rIter = _rList.MakeIterator() ;
  _fIter = _fList.MakeIterator() ;
  _mIter = _mList.MakeIterator() ;
  _yIter = _yList.MakeIterator() ;
  _pIter = _pList.MakeIterator() ;
}


RooCmdConfig::RooCmdConfig(const RooCmdConfig& other) 
{
  // Copy constructor
  _name   = other._name ;
  _verbose = other._verbose ;
  _error = other._error ;
  _allowUndefined = other._allowUndefined ;

  _iIter = _iList.MakeIterator() ;
  _dIter = _dList.MakeIterator() ;
  _sIter = _sList.MakeIterator() ;
  _oIter = _oList.MakeIterator() ;
  _rIter = _rList.MakeIterator() ;
  _fIter = _fList.MakeIterator() ;
  _mIter = _mList.MakeIterator() ;
  _yIter = _yList.MakeIterator() ;
  _pIter = _pList.MakeIterator() ;

  other._iIter->Reset() ;
  RooInt* ri ;
  while(ri=(RooInt*)other._iIter->Next()) {
    _iList.Add(ri->Clone()) ;
  }

  other._dIter->Reset() ;
  RooDouble* rd ;
  while(rd=(RooDouble*)other._dIter->Next()) {
    _dList.Add(rd->Clone()) ;
  }

  other._sIter->Reset() ;
  RooStringVar* rs ;
  while(rs=(RooStringVar*)other._sIter->Next()) {
    _sList.Add(rs->Clone()) ;
  }

  other._oIter->Reset() ;
  RooTObjWrap* os ;
  while(os=(RooTObjWrap*)other._oIter->Next()) {
    _oList.Add(os->Clone()) ;
  }

  other._rIter->Reset() ;
  TObjString* rr ;
  while(rr=(TObjString*)other._rIter->Next()) {
    _rList.Add(rr->Clone()) ;
  }

  other._fIter->Reset() ;
  TObjString* ff ;
  while(ff=(TObjString*)other._fIter->Next()) {
    _fList.Add(ff->Clone()) ;
  }

  other._mIter->Reset() ;
  TObjString* mm ;
  while(mm=(TObjString*)other._mIter->Next()) {
    _mList.Add(mm->Clone()) ;
  }

  other._yIter->Reset() ;
  TObjString* yy ;
  while(yy=(TObjString*)other._yIter->Next()) {
    _yList.Add(yy->Clone()) ;
  }

  other._pIter->Reset() ;
  TObjString* pp ;
  while(pp=(TObjString*)other._pIter->Next()) {
    _pList.Add(pp->Clone()) ;
  }

}



RooCmdConfig::~RooCmdConfig()
{
  // Destructor 
  delete _iIter ;
  delete _dIter ;
  delete _sIter ;
  delete _oIter ;
  delete _rIter ;
  delete _fIter ;
  delete _mIter ;
  delete _yIter ;
  delete _pIter ;

  _iList.Delete() ;
  _dList.Delete() ;
  _sList.Delete() ;
  _oList.Delete() ;
  _rList.Delete() ;
  _fList.Delete() ;
  _mList.Delete() ;
  _yList.Delete() ;
  _pList.Delete() ;
}



void RooCmdConfig::defineRequiredArgs(const char* argName1, const char* argName2,
				      const char* argName3, const char* argName4,
				      const char* argName5, const char* argName6,
				      const char* argName7, const char* argName8) 
{
  if (argName1) _rList.Add(new TObjString(argName1)) ;
  if (argName2) _rList.Add(new TObjString(argName2)) ;
  if (argName3) _rList.Add(new TObjString(argName3)) ;
  if (argName4) _rList.Add(new TObjString(argName4)) ;
  if (argName5) _rList.Add(new TObjString(argName5)) ;
  if (argName6) _rList.Add(new TObjString(argName6)) ;
  if (argName7) _rList.Add(new TObjString(argName7)) ;
  if (argName8) _rList.Add(new TObjString(argName8)) ;
}


const char* RooCmdConfig::missingArgs() const 
{
  static TString ret ;
  ret="" ;

  _rIter->Reset() ;
  TObjString* s ;
  Bool_t first(kTRUE) ;
  while(s=(TObjString*)_rIter->Next()) {
    if (first) {
      first=kFALSE ;
    } else {
      ret.Append(", ") ;
    }
    ret.Append(s->String()) ;
  }

  return ret.Length() ? ret.Data() : 0 ;
}



void RooCmdConfig::defineDependency(const char* refArgName, const char* neededArgName) 
{
  TNamed* dep = new TNamed(refArgName,neededArgName) ;
  _yList.Add(dep) ;
}



void RooCmdConfig::defineMutex(const char* argName1, const char* argName2) 
{
  TNamed* mutex1 = new TNamed(argName1,argName2) ;
  TNamed* mutex2 = new TNamed(argName2,argName1) ;
  _mList.Add(mutex1) ;
  _mList.Add(mutex2) ;
}



Bool_t RooCmdConfig::defineInt(const char* name, const char* argName, Int_t intNum, Int_t defVal)
{
  if (_iList.FindObject(name)) {
    cout << "RooCmdConfig::defintInt: name '" << name << "' already defined" << endl ;
    return kTRUE ;
  }

  RooInt* ri = new RooInt(defVal) ;
  ri->SetName(name) ;
  ri->SetTitle(argName) ;
  ri->SetUniqueID(intNum) ;
  
  _iList.Add(ri) ;
  return kFALSE ;
}



Bool_t RooCmdConfig::defineDouble(const char* name, const char* argName, Int_t doubleNum, Double_t defVal) 
{
  if (_dList.FindObject(name)) {
    cout << "RooCmdConfig::defineDouble: name '" << name << "' already defined" << endl ;
    return kTRUE ;
  }

  RooDouble* rd = new RooDouble(defVal) ;
  rd->SetName(name) ;
  rd->SetTitle(argName) ;
  rd->SetUniqueID(doubleNum) ;
  
  _dList.Add(rd) ;
  return kFALSE ;
}



Bool_t RooCmdConfig::defineString(const char* name, const char* argName, Int_t stringNum, const char* defVal) 
{
  if (_sList.FindObject(name)) {
    cout << "RooCmdConfig::defineString: name '" << name << "' already defined" << endl ;
    return kTRUE ;
  }

  RooStringVar* rs = new RooStringVar(name,argName,defVal) ;
  rs->SetUniqueID(stringNum) ;
  
  _sList.Add(rs) ;
  return kFALSE ;
}



Bool_t RooCmdConfig::defineObject(const char* name, const char* argName, Int_t setNum, const TObject* defVal) 
{
  if (_oList.FindObject(name)) {
    cout << "RooCmdConfig::defineObject: name '" << name << "' already defined" << endl ;
    return kTRUE ;
  }

  RooTObjWrap* os = new RooTObjWrap((TObject*)defVal) ;
  os->SetName(name) ;
  os->SetTitle(argName) ;
  os->SetUniqueID(setNum) ;
  
  _oList.Add(os) ;
  return kFALSE ;
}


void RooCmdConfig::print()
{
  // Find registered integer fields for this opcode 
  _iIter->Reset() ;
  RooInt* ri ;
  while(ri=(RooInt*)_iIter->Next()) {
    cout << ri->GetName() << "[Int_t] = " << *ri << endl ;
  }

  // Find registered double fields for this opcode 
  _dIter->Reset() ;
  RooDouble* rd ;
  while(rd=(RooDouble*)_dIter->Next()) {
    cout << rd->GetName() << "[Double_t] = " << *rd << endl ;
  }

  // Find registered string fields for this opcode 
  _sIter->Reset() ;
  RooStringVar* rs ;
  while(rs=(RooStringVar*)_sIter->Next()) {
    cout << rs->GetName() << "[string] = \"" << rs->getVal() << "\"" << endl ;
  }

  // Find registered argset fields for this opcode 
  _oIter->Reset() ;
  RooTObjWrap* ro ;
  while(ro=(RooTObjWrap*)_oIter->Next()) {
    cout << ro->GetName() << "[TObject] = " ; 
    if (ro->obj()) {
      cout << ro->obj()->GetName() << endl ;
    } else {

      cout << "(null)" << endl ;
    }
  }
}


Bool_t RooCmdConfig::process(TList& argList) 
{
  Bool_t ret(kFALSE) ;
  TIterator* iter = argList.MakeIterator() ;
  RooCmdArg* arg ;
  while(arg=(RooCmdArg*)iter->Next()) {
    ret |= process(*arg) ;
  }
  delete iter ;
  return ret ;
}


Bool_t RooCmdConfig::process(const RooCmdArg& arg) 
{
  // Retrive command code
  const char* opc = arg.opcode() ;

  // Ignore empty commands
  if (!opc) return kFALSE ;

  // Check if not forbidden
  if (_fList.FindObject(opc)) {
    cout << _name << " ERROR: argument " << opc << " not allowed in this context" << endl ;
    _error = kTRUE ;
    return kTRUE ;
  }

  // Check if this code generates any dependencies
  TObject* dep = _yList.FindObject(opc) ;
  if (dep) {
    // Dependent command found, add to required list if not already processed
    if (!_pList.FindObject(dep->GetTitle())) {
      _rList.Add(new TObjString(dep->GetTitle())) ;
      if (_verbose) {
	cout << "RooCmdConfig::process: " << opc << " has unprocessed dependent " << dep->GetTitle() 
	     << ", adding to required list" << endl ;
      }
    } else {
      if (_verbose) {
	cout << "RooCmdConfig::process: " << opc << " dependent " << dep->GetTitle() << " is already processed" << endl ;
      }
    }
  }

  // Check for mutexes
  TObject * mutex = _mList.FindObject(opc) ;
  if (mutex) {
    if (_verbose) {
      cout << "RooCmdConfig::process: " << opc << " excludes " << mutex->GetTitle() 
	   << ", adding to forbidden list" << endl ;
    }    
    _fList.Add(new TObjString(mutex->GetTitle())) ;
  }


  Bool_t anyField(kFALSE) ;

  // Find registered integer fields for this opcode 
  _iIter->Reset() ;
  RooInt* ri ;
  while(ri=(RooInt*)_iIter->Next()) {
    if (!TString(opc).CompareTo(ri->GetTitle())) {
      *ri = arg.getInt(ri->GetUniqueID()) ;
      anyField = kTRUE ;
      if (_verbose) {
	cout << "RooCmdConfig::process " << ri->GetName() << "[Int_t]" << " set to " << *ri << endl ;
      }
    }
  }

  // Find registered double fields for this opcode 
  _dIter->Reset() ;
  RooDouble* rd ;
  while(rd=(RooDouble*)_dIter->Next()) {
    if (!TString(opc).CompareTo(rd->GetTitle())) {
      *rd = arg.getDouble(rd->GetUniqueID()) ;
      anyField = kTRUE ;
      if (_verbose) {
	cout << "RooCmdConfig::process " << rd->GetName() << "[Double_t]" << " set to " << *rd << endl ;
      }
    }
  }

  // Find registered string fields for this opcode 
  _sIter->Reset() ;
  RooStringVar* rs ;
  while(rs=(RooStringVar*)_sIter->Next()) {
    if (!TString(opc).CompareTo(rs->GetTitle())) {
      rs->setVal(arg.getString(rs->GetUniqueID())) ;
      anyField = kTRUE ;
      if (_verbose) {
	cout << "RooCmdConfig::process " << rs->GetName() << "[string]" << " set to " << rs->getVal() << endl ;
      }
    }
  }

  // Find registered dataset fields for this opcode 
  _oIter->Reset() ;
  RooTObjWrap* os ;
  while(os=(RooTObjWrap*)_oIter->Next()) {
    if (!TString(opc).CompareTo(os->GetTitle())) {
      os->setObj((TObject*)arg.getObject(os->GetUniqueID())) ;
      anyField = kTRUE ;
      if (_verbose) {
	cout << "RooCmdConfig::process " << os->GetName() << "[TObject]" << " set to " ;
	if (os->obj()) {
	  cout << os->obj()->GetName() << endl ;
	} else {
	  cout << "(null)" << endl ;
	}
      }
    }
  }

  if (!anyField && !_allowUndefined) {
    cout << _name << " ERROR: unrecognized command: " << opc << endl ;
  }


  // Remove command from required-args list (if it was there)
  TObject* obj = _rList.FindObject(opc) ;
  if (obj) {
    _rList.Remove(obj) ;
  }

  // Add command the processed list
  TNamed *pcmd = new TNamed(opc,opc) ;
  _pList.Add(pcmd) ;

  return (anyField||_allowUndefined)?kFALSE:kTRUE ;
}
  


Int_t RooCmdConfig::getInt(const char* name, Int_t defVal) 
{
  RooInt* ri = (RooInt*) _iList.FindObject(name) ;
  return ri ? (Int_t)(*ri) : defVal ;
}



Double_t RooCmdConfig::getDouble(const char* name, Double_t defVal) 
{
  RooDouble* rd = (RooDouble*) _dList.FindObject(name) ;
  return rd ? (Double_t)(*rd) : defVal ;
}



const char* RooCmdConfig::getString(const char* name, const char* defVal) 
{
  RooStringVar* rs = (RooStringVar*) _sList.FindObject(name) ;
  return rs ? (const char*)rs->getVal() : defVal ;
}



const TObject* RooCmdConfig::getObject(const char* name, const TObject* defVal) 
{
  RooTObjWrap* ro = (RooTObjWrap*) _oList.FindObject(name) ;
  return ro ? ro->obj() : defVal ;
}


Bool_t RooCmdConfig::ok(Bool_t verbose) const 
{ 
  if (_rList.GetSize()==0 && !_error) return kTRUE ;

  if (verbose) {
    const char* margs = missingArgs() ;
    if (margs) {
      cout << _name << " ERROR: missing arguments: " << margs << endl ;
    } else {
      cout << _name << " ERROR: illegal combination of arguments and/or missing arguments" << endl ;
    }
  }
  return kFALSE ;
}


void RooCmdConfig::stripCmdList(TList& cmdList, const char* cmdsToPurge) 
{
  // Strip command names listed (comma separated) in cmdsToPurge from cmdList

  // Sanity check
  if (!cmdsToPurge) return ;
  
  // Copy command list for parsing
  char buf[1024] ;
  strcpy(buf,cmdsToPurge) ;
  
  char* name = strtok(buf,",") ;
  while(name) {
    TObject* cmd = cmdList.FindObject(name) ;
    if (cmd) cmdList.Remove(cmd) ;
    name = strtok(0,",") ;
  }

}
