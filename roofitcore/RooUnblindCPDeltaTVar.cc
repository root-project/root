/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooUnblindCPDeltaTVar.cc,v 1.2 2001/03/29 01:59:10 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooUnblindCPDeltaTVar.hh"
#include "RooFitCore/RooArgSet.hh"


ClassImp(RooUnblindCPDeltaTVar)
;


RooUnblindCPDeltaTVar::RooUnblindCPDeltaTVar() : _blindEngine("") 
{
}


RooUnblindCPDeltaTVar::RooUnblindCPDeltaTVar(const char *name, const char *title,
					     const char *blindString,
					     RooAbsReal& deltat, RooAbsCategory& tag, 
					     RooAbsCategory& blindState)
  : RooAbsReal(name,title), _blindEngine(blindString), _deltat(&deltat), 
    _tag(&tag), _state(&blindState)
{  
  addServer(deltat) ;
  addServer(tag) ;
  addServer(blindState) ;
}


RooUnblindCPDeltaTVar::RooUnblindCPDeltaTVar(const char* name, const RooUnblindCPDeltaTVar& other) : 
  RooAbsReal(name, other), _blindEngine(other._blindEngine), _deltat(other._deltat),
  _tag(other._tag), _state(other._state)
{
}


RooUnblindCPDeltaTVar::~RooUnblindCPDeltaTVar() 
{
}


Double_t RooUnblindCPDeltaTVar::evaluate() const
{
  if (_state->getIndex()==0) {
    // Blinding not active for this event
    return _deltat->getVal() ;
  } else {
    // Blinding active for this event
    return _blindEngine.UnHideDeltaZ(_deltat->getVal(),_tag->getIndex());
  }
}


Double_t RooUnblindCPDeltaTVar::getVal() const
{
  // Call parent class implementation
  return RooAbsReal::getVal() ;
}


Bool_t RooUnblindCPDeltaTVar::isValid() const
{
  return isValid(getVal()) ;
}


Bool_t RooUnblindCPDeltaTVar::isValid(Double_t value, Bool_t verbose) const
{
  return kTRUE ;
}



Bool_t RooUnblindCPDeltaTVar::redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll) 
{
  RooAbsReal* newDeltaT = (RooAbsReal*) newServerList.find(_deltat->GetName()) ;
  if (!newDeltaT) {
    if (mustReplaceAll) {
      cout << "RooUnblindCPDeltaTVar::redirectServersHook(" << GetName() 
	   << "): cannot find server named " << _deltat->GetName() << endl ;
      return kTRUE ;
    }
  } else {
    _deltat = newDeltaT ;
  }

  RooAbsCategory* newTag = (RooAbsCategory*) newServerList.find(_tag->GetName()) ;
  if (!newTag) {
    if (mustReplaceAll) {
      cout << "RooUnblindCPDeltaTVar::redirectServersHook(" << GetName() 
	   << "): cannot find server named " << _tag->GetName() << endl ;
      return kTRUE ;
    }
  } else {
    _tag = newTag ;
  }

  RooAbsCategory* newState = (RooAbsCategory*) newServerList.find(_state->GetName()) ;
  if (!newState) {
    if (mustReplaceAll) {
      cout << "RooUnblindCPDeltaTVar::redirectServersHook(" << GetName() 
	   << "): cannot find server named " << _state->GetName() << endl ;
      return kTRUE ;
    }
  } else {
    _state = newState ;
  }

  return kFALSE ;
}



void RooUnblindCPDeltaTVar::printToStream(ostream& os, PrintOption opt) const
{
  // Print current value and definition of formula
  os << "RooUnblindCPDeltaTVar: " << GetName() << " : (value hidden) deltat=" 
     << _deltat->GetName() << ", tag=" << _tag->GetName() ;
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



