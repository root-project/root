/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooUnblindCPAsymVar.cc,v 1.1 2001/03/29 01:06:44 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooUnblindCPAsymVar.hh"
#include "RooFitCore/RooArgSet.hh"


ClassImp(RooUnblindCPAsymVar)
;


RooUnblindCPAsymVar::RooUnblindCPAsymVar() : _blindEngine("") 
{
}


RooUnblindCPAsymVar::RooUnblindCPAsymVar(const char *name, const char *title,
					     const char *blindString, RooAbsReal& cpasym)
  : RooAbsReal(name,title), _blindEngine(blindString), _asym(&cpasym) 
{  
  addServer(cpasym) ;
}


RooUnblindCPAsymVar::RooUnblindCPAsymVar(const RooUnblindCPAsymVar& other) : 
  RooAbsReal(other), _blindEngine(other._blindEngine), _asym(other._asym)
{
}


RooUnblindCPAsymVar::~RooUnblindCPAsymVar() 
{
}


Double_t RooUnblindCPAsymVar::evaluate() const
{
  return _blindEngine.UnHideAsym(_asym->getVal());
}


Double_t RooUnblindCPAsymVar::getVal() const
{
  // Call parent class implementation
  return RooAbsReal::getVal() ;
}


Bool_t RooUnblindCPAsymVar::isValid() const
{
  return isValid(getVal()) ;
}


Bool_t RooUnblindCPAsymVar::isValid(Double_t value, Bool_t verbose) const
{
  return kTRUE ;
}


Bool_t RooUnblindCPAsymVar::redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll) 
{
  RooAbsReal* newAsym = (RooAbsReal*) newServerList.find(_asym->GetName()) ;
  if (!newAsym) {
    if (mustReplaceAll) {
      cout << "RooUnblindCPDeltaTVar::redirectServersHook(" << GetName() 
	   << "): cannot find server named " << _asym->GetName() << endl ;
      return kTRUE ;
    }
  } else {
    _asym = newAsym ;
  }

  return kFALSE ;
}


void RooUnblindCPAsymVar::printToStream(ostream& os, PrintOption opt) const
{
  // Print current value and definition of formula
  os << "RooUnblindCPAsymVar: " << GetName() << " : (value hidden) asym=" 
     << _asym->GetName() ;
  if(!_unit.IsNull()) os << ' ' << _unit;
  printAttribList(os) ;
  os << endl ;
} 


Bool_t RooUnblindCPAsymVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  cout << "RooUnblindCPAsymVar::readFromStream(" << GetName() << "): not allowed" << endl ;
  return kTRUE ;
}


void RooUnblindCPAsymVar::writeToStream(ostream& os, Bool_t compact) const
{
  cout << "RooUnblindCPAsymVar::writeToStream(" << GetName() << "): not allowed" << endl ;
}



