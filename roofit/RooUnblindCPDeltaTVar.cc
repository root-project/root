/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooUnblindCPDeltaTVar.cc,v 1.5 2001/08/03 18:13:02 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooArgSet.hh"
#include "RooFitModels/RooUnblindCPDeltaTVar.hh"


ClassImp(RooUnblindCPDeltaTVar)
;


RooUnblindCPDeltaTVar::RooUnblindCPDeltaTVar() : _blindEngine("") 
{
}


RooUnblindCPDeltaTVar::RooUnblindCPDeltaTVar(const char *name, const char *title,
					     const char *blindString,
					     RooAbsRealLValue& deltat, RooAbsCategory& tag, 
					     RooAbsCategory& blindState)
  : RooAbsRealLValue(name,title), _blindEngine(blindString), _deltat("deltat","Delta t",this,deltat), 
    _tag("tag","CP Tag",this,tag), _state("state","Blinding state",this,blindState)
{  
}


RooUnblindCPDeltaTVar::RooUnblindCPDeltaTVar(const RooUnblindCPDeltaTVar& other, const char* name) : 
  RooAbsRealLValue(other, name), _blindEngine(other._blindEngine), _deltat("deltat",this,other._deltat),
  _tag("tag",this,other._tag), _state("state",this,other._state)
{
  setVal(other.getVal()) ;
}


RooUnblindCPDeltaTVar::~RooUnblindCPDeltaTVar() 
{
}


Double_t RooUnblindCPDeltaTVar::evaluate() const
{
  if ((const char*)_state=="Blind") {
    // Blinding active for this event
    return _blindEngine.UnHideDeltaZ(_deltat,_tag);
  } else {
    // Blinding not active for this event
    return _deltat ;
  }
}


void RooUnblindCPDeltaTVar::setVal(Double_t value) 
{
  if ((const char*)_state=="Blind") {
    // Blinding active for this event
    ((RooAbsRealLValue&)_deltat.arg()) = _blindEngine.HideDeltaZ(value,_tag);
  } else {
    // Blinding not active for this event
    ((RooAbsRealLValue&)_deltat.arg()) = value ;
  }
}



Double_t RooUnblindCPDeltaTVar::getFitMin() const 
{
  return _deltat.min() ;
}



Double_t RooUnblindCPDeltaTVar::getFitMax() const 
{
  return _deltat.max() ;
}




Bool_t RooUnblindCPDeltaTVar::isValid() const
{
  return isValid(getVal()) ;
}


Bool_t RooUnblindCPDeltaTVar::isValid(Double_t value, Bool_t verbose) const
{
  return kTRUE ;
}


void RooUnblindCPDeltaTVar::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print current value and definition of formula
  os << indent << "RooUnblindCPDeltaTVar: " << GetName() << " : (value hidden) deltat=" 
     << _deltat.arg().GetName() << ", tag=" << _tag.arg().GetName() ;
  if(!_unit.IsNull()) os << ' ' << _unit;
  printAttribList(os) ;
  os << endl ;
} 


Bool_t RooUnblindCPDeltaTVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  cout << "RooUnblindCPDeltaTVar::readFromStream(" << GetName() << "): not allowed" << endl ;
  return kTRUE ;
}


void RooUnblindCPDeltaTVar::writeToStream(ostream& os, Bool_t compact) const
{
  cout << "RooUnblindCPDeltaTVar::writeToStream(" << GetName() << "): not allowed" << endl ;
}



