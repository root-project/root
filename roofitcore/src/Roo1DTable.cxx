/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: Roo1DTable.cc,v 1.5 2001/05/17 00:43:14 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// Roo1DTable implements a one-dimensional table. A table is the category
// equivalent of a plot. To create a table use the RooDataSet::table method.

#include <iostream.h>
#include <iomanip.h>
#include "TString.h"
#include "RooFitCore/Roo1DTable.hh"

ClassImp(Roo1DTable)

Roo1DTable::Roo1DTable(const char *name, const char *title, const RooAbsCategory& cat) : 
  RooTable(name,title), _nOverflow(0)
{
  // Create an empty table from abstract category. The number of table entries and 
  // their names are taken from the category state labels at the time of construction,
  // but not reference to the category is retained after the construction phase.
  // Use fill() to fill the table.

  //Take types from reference category
  Int_t nbin(0) ;
  TIterator* tIter = cat.typeIterator() ;
  RooCatType* type ;
  while (type = (RooCatType*)tIter->Next()) {
    _types.Add(new RooCatType(*type)) ;
    nbin++ ;
  }
  delete tIter ;

  // Create counter array and initialize
  _count = new Int_t[nbin] ;
  for (int i=0 ; i<nbin ; i++) _count[i] = 0 ;
}



Roo1DTable::Roo1DTable(const Roo1DTable& other) : RooTable(other), _nOverflow(other._nOverflow) 
{  
  // Copy constructor

  //Take types from reference category
  RooCatType* type ;
  Int_t nbin(0) ;
  for (int i=0 ; i<other._types.GetEntries() ; i++) {
    _types.Add(new RooCatType(*(RooCatType*)other._types.At(i))) ;
    nbin++ ;
  }

  // Create counter array and initialize
  _count = new Int_t[nbin] ;
  for (int i=0 ; i<nbin ; i++) _count[i] = other._count[i] ;
}


Roo1DTable::~Roo1DTable()
{
  // Destructor

  // We own the contents of the object array
  _types.Delete() ;
  delete[] _count ;
}


void Roo1DTable::fill(RooAbsCategory& cat) 
{
  // Increment the counter of the table slot with
  // the name corresponding to that of the current 
  // category state. If the current category state
  // matches no table slot name, the table overflow
  // counter is incremented.

  Bool_t found(kFALSE) ;
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _types.At(i) ;
    if (cat.getIndex()==entry->getVal()) {
      _count[i]++ ;
      found=kTRUE ;
    }
  }  

  if (!found) _nOverflow++ ;
}



void Roo1DTable::printToStream(ostream& os, PrintOption opt, TString indent) const 
{
  // Print the formateed table contents on the given stream

  os << endl ;
  os << "  Table " << GetName() << " : " << GetTitle() << endl ;

  // Determine maximum label and count width
  Int_t labelWidth(0) ;
  Int_t maxCount(1) ;
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _types.At(i) ;
    labelWidth=strlen(entry->GetName())>labelWidth
              ?strlen(entry->GetName()):labelWidth ;    
    maxCount=_count[i]>maxCount?_count[i]:maxCount ;
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
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _types.At(i) ;
    os << "  | " << setw(labelWidth) << entry->GetName() << " | " << setw(countWidth) << _count[i] << " |" << endl ;
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

