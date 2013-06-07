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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// Roo1DTable implements a one-dimensional table. A table is the category
// equivalent of a plot. To create a table use the RooDataSet::table method.
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include <iomanip>
#include "TString.h"
#include "TMath.h"
#include "Roo1DTable.h"
#include "RooMsgService.h"
#include "TClass.h"

using namespace std ;

ClassImp(Roo1DTable)


//_____________________________________________________________________________
Roo1DTable::Roo1DTable(const char *name, const char *title, const RooAbsCategory& cat) : 
  RooTable(name,title), _total(0), _nOverflow(0)
{
  // Create an empty table from abstract category. The number of table entries and 
  // their names are taken from the category state labels at the time of construction,
  // but not reference to the category is retained after the construction phase.
  // Use fill() to fill the table.

  //Take types from reference category
  Int_t nbin=0 ;
  TIterator* tIter = cat.typeIterator() ;
  RooCatType* type ;
  while (((type = (RooCatType*)tIter->Next()))) {
    _types.Add(new RooCatType(*type)) ;
    nbin++ ;
  }
  delete tIter ;

  // Create counter array and initialize
  _count.resize(nbin) ;
  for (int i=0 ; i<nbin ; i++) _count[i] = 0 ;
}



//_____________________________________________________________________________
Roo1DTable::Roo1DTable(const Roo1DTable& other) : 
  RooTable(other), _count(other._count), _total(other._total), _nOverflow(other._nOverflow)
{  
  // Copy constructor

  // Take types from reference category

  int i;
  for (i=0 ; i<other._types.GetEntries() ; i++) {
    _types.Add(new RooCatType(*(RooCatType*)other._types.At(i))) ;
  }

}



//_____________________________________________________________________________
Roo1DTable::~Roo1DTable()
{
  // Destructor

  // We own the contents of the object array
  _types.Delete() ;
}



//_____________________________________________________________________________
void Roo1DTable::fill(RooAbsCategory& cat, Double_t weight) 
{
  // Increment the counter of the table slot with the name
  // corresponding to that of the current category state. If the
  // current category state matches no table slot name, the table
  // overflow counter is incremented.

  if (weight==0) return ;

  _total += weight ;

  //Bool_t found(kFALSE) ;
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _types.At(i) ;
    if (cat.getIndex()==entry->getVal()) {
      _count[i] += weight ; ;
      //found=kTRUE ;
      return;
    }
  }  

  //if (!found) {
  _nOverflow += weight ;
  //}
}



//_____________________________________________________________________________
void Roo1DTable::printName(ostream& os) const 
{
  // Print the name of the table
  os << GetName() ;
}



//_____________________________________________________________________________
void Roo1DTable::printTitle(ostream& os) const 
{
  // Print the title of the table
  os << GetTitle() ;
}



//_____________________________________________________________________________
void Roo1DTable::printClassName(ostream& os) const 
{
  // Print the class name of the table
  os << IsA()->GetName() ;
}



//_____________________________________________________________________________
void Roo1DTable::printValue(ostream& os) const 
{
  // Print the table value, i.e. the contents, in 'inline' format
  os << "(" ;
  for (Int_t i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _types.At(i) ;
    if (_count[i]>0) {
      if (i>0) {
	os << "," ;
      }
      os << entry->GetName() << "=" << _count[i] ;
    }
  }
  os << ")" ;
}




//_____________________________________________________________________________
Int_t Roo1DTable::defaultPrintContents(Option_t* /*opt*/) const 
{
  // Define default contents to print
  return kName|kClassName|kValue|kArgs ;
}



//_____________________________________________________________________________
void Roo1DTable::printMultiline(ostream& os, Int_t /*contents*/, Bool_t verbose, TString indent) const 
{
  // Print the formatted table contents on the given stream
  
  os << indent << endl ;
  os << indent << "  Table " << GetName() << " : " << GetTitle() << endl ;

  // Determine maximum label and count width
  Int_t labelWidth(0) ;
  Double_t maxCount(1) ;

  int i;
  for (i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _types.At(i) ;

    // Disable warning about a signed/unsigned mismatch by MSCV 6.0 by
    // using the lwidth temporary.
    Int_t lwidth = strlen(entry->GetName());
    labelWidth = lwidth > labelWidth ? lwidth : labelWidth;
    maxCount=_count[i]>maxCount?_count[i]:maxCount ;
  }
  // Adjust formatting if overflow field will be present
  if (_nOverflow>0) {
    labelWidth=labelWidth>8?labelWidth:8 ;
    maxCount=maxCount>_nOverflow?maxCount:_nOverflow ;
  }

  // Header
  Int_t countWidth=((Int_t)log10(maxCount))+1 ;
  os << indent << "  +-" << setw(labelWidth) << setfill('-') << "-" << "-+-" << setw(countWidth) << "-" << "-+" << endl ;
  os << setfill(' ') ;

  // Contents
  for (i=0 ; i<_types.GetEntries() ; i++) {
    RooCatType* entry = (RooCatType*) _types.At(i) ;
    if (_count[i]>0 || verbose) {
      os << "  | " << setw(labelWidth) << entry->GetName() << " | " << setw(countWidth) << _count[i] << " |" << endl ;
    }
  }

  // Overflow field
  if (_nOverflow) {
    os << indent << "  +-" << setw(labelWidth) << setfill('-') << "-" << "-+-" << setw(countWidth) << "-" << "-+" << endl ;
    os << indent << "  | " << "Overflow" << " | " << setw(countWidth) << _nOverflow << " |" << endl ;    
  }

  // Footer
  os << indent << "  +-" << setw(labelWidth) << setfill('-') << "-" << "-+-" << setw(countWidth) << "-" << "-+" << endl ;
  os << setfill(' ') ;
  os << indent << endl ;
}



//_____________________________________________________________________________
Double_t Roo1DTable::get(const char* label, Bool_t silent) const 
{
  // Return the table entry named 'label'. Zero is returned if given
  // label doesn't occur in table.


  TObject* cat = _types.FindObject(label) ;
  if (!cat) {
    if (!silent) {
      coutE(InputArguments) << "Roo1DTable::get: ERROR: no such entry: " << label << endl ;
    }
    return 0 ;
  }
  return _count[_types.IndexOf(cat)] ;
}



//_____________________________________________________________________________
Double_t Roo1DTable::get(const int index, Bool_t silent) const 
{
  // Return the table entry named 'label'. Zero is returned if given
  // label doesn't occur in table.

  const RooCatType* cat = 0;
  int i = 0;
  for (; i < _types.GetEntries(); ++i) {
     cat = static_cast<const RooCatType*>(_types[i]);
     if (cat->getVal() == index) {
        break;
     } else {
        cat = 0;
     }
  }
  if (!cat) {
    if (!silent) {
      coutE(InputArguments) << "Roo1DTable::get: ERROR: no such entry: " << index << endl ;
    }
    return 0 ;
  }
  return _count[i] ;
}



//_____________________________________________________________________________
Double_t Roo1DTable::getOverflow() const 
{
  // Return the number of overflow entries in the table.

  return _nOverflow ;
}



//_____________________________________________________________________________
Double_t Roo1DTable::getFrac(const char* label, Bool_t silent) const 
{
  // Return the fraction of entries in the table contained in the slot named 'label'. 
  // The normalization includes the number of overflows.
  // Zero is returned if given label doesn't occur in table.   

  if (_total) {
    return get(label,silent) / _total ;
  } else {
    if (!silent) coutW(Contents) << "Roo1DTable::getFrac: WARNING table empty, returning 0" << endl ;
    return 0. ;
  }
}



//_____________________________________________________________________________
Double_t Roo1DTable::getFrac(const int index, Bool_t silent) const 
{
  // Return the fraction of entries in the table contained in the slot named 'label'. 
  // The normalization includes the number of overflows.
  // Zero is returned if given label doesn't occur in table.   

  if (_total) {
    return get(index, silent) / _total ;
  } else {
    if (!silent) coutW(Contents) << "Roo1DTable::getFrac: WARNING table empty, returning 0" << endl ;
    return 0. ;
  }
}



//_____________________________________________________________________________
Bool_t Roo1DTable::isIdentical(const RooTable& other) 
{
  // Return true if table is identical in contents to given reference table

  const Roo1DTable* other1d = &dynamic_cast<const Roo1DTable&>(other) ;

  if (!other1d) {
    return kFALSE ;
  }

  int i;
  for (i=0 ; i<_types.GetEntries() ; i++) {
    // RooCatType* entry = (RooCatType*) _types.At(i) ;        
    if (_count[i] != other1d->_count[i]) {
      return kFALSE ;
    }
  }
  return kTRUE ;
}
