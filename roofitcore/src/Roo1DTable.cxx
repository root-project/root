/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include <iomanip.h>
#include "TString.h"
#include "RooFitCore/Roo1DTable.hh"

ClassImp(Roo1DTable)

Roo1DTable::Roo1DTable(const char *name, const char *title, RooAbsCategory& cat) : RooTable(name,title), _nOverflow(0)
{
  //Take types from reference category
  TIterator* tIter = cat.typeIterator() ;
  RooCatType* type ;
  while (type = (RooCatType*)tIter->Next()) {
    _contents.Add(new RooCatType(type->GetName(),0)) ;
  }
  delete tIter ;
}



Roo1DTable::Roo1DTable(const Roo1DTable& other) : RooTable(other), _nOverflow(other._nOverflow) 
{  
  //Take types from reference category
  RooCatType* type ;
  for (int i=0 ; i<other._contents.GetEntries() ; i++) {
    _contents.Add(new RooCatType(*(RooCatType*)other._contents.At(i))) ;
  }
}


Roo1DTable::~Roo1DTable()
{
  // We own the contents of the object array
  _contents.Delete() ;
}


void Roo1DTable::fill(RooAbsCategory& cat) 
{
  Bool_t found(kFALSE) ;
  for (int i=0 ; i<_contents.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _contents.At(i) ;
    if (!TString(cat.getLabel()).CompareTo(entry->GetName())) {
      entry->setVal(entry->getVal()+1) ;
      found=kTRUE ;
    }
  }  

  if (!found) _nOverflow++ ;
}



void Roo1DTable::printToStream(ostream& os, PrintOption opt=Standard) 
{
  os << endl ;
  os << "  Table " << GetName() << " : " << GetTitle() << endl ;

  // Determine maximum label and count width
  Int_t labelWidth(0) ;
  Int_t maxCount(1) ;
  for (int i=0 ; i<_contents.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _contents.At(i) ;
    labelWidth=strlen(entry->GetName())>labelWidth
              ?strlen(entry->GetName()):labelWidth ;    
    maxCount=entry->getVal()>maxCount?entry->getVal():maxCount ;
  }
  // Adjust formatting if overflow field will be present
  if (_nOverflow>0) {
    labelWidth=labelWidth>8?labelWidth:8 ;
    maxCount=maxCount>_nOverflow?maxCount:_nOverflow ;
  }

  // Header
  Int_t countWidth=((Int_t)log10(maxCount))+1 ;
  os << "  +-" << setw(labelWidth) << setfill('-') << "-" << "-+-" << setw(countWidth) << "-" << "-+" << endl ;
  os << setfill(' ') ;

  // Contents
  for (int i=0 ; i<_contents.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _contents.At(i) ;
    os << "  | " << setw(labelWidth) << entry->GetName() << " | " << setw(countWidth) << entry->getVal() << " |" << endl ;
  }

  // Overflow field
  if (_nOverflow) {
    os << "  +-" << setw(labelWidth) << setfill('-') << "-" << "-+-" << setw(countWidth) << "-" << "-+" << endl ;
    os << "  | " << "Overflow" << " | " << setw(countWidth) << _nOverflow << " |" << endl ;    
  }

  // Footer
  os << "  +-" << setw(labelWidth) << setfill('-') << "-" << "-+-" << setw(countWidth) << "-" << "-+" << endl ;
  os << setfill(' ') ;
  os << endl ;
}

