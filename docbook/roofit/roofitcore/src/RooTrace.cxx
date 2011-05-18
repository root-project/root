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
// Class RooTrace controls the memory tracing hooks in all RooFit
// objects. When tracing is active, a table of live RooFit objects
// is kept that can be queried at any time. In verbose mode, messages
// are printed in addition at the construction and destruction of
// each object.
// END_HTML
//

#include "RooFit.h"

#include "RooTrace.h"
#include "RooAbsArg.h"
#include "Riostream.h"

#include <iomanip>



ClassImp(RooTrace)
;


Bool_t RooTrace::_active(kFALSE) ;
Bool_t RooTrace::_verbose(kFALSE) ;
RooLinkedList RooTrace::_list ;
RooLinkedList RooTrace::_markList ;



//_____________________________________________________________________________
void RooTrace::create(const TObject* obj) 
{ 
  // Register creation of object 'obj' 

  if (_active) create2(obj) ; 
}


//_____________________________________________________________________________
void RooTrace::destroy(const TObject* obj) 
{ 
  // Register deletion of object 'obj'

  if (_active) destroy2(obj) ; 
}


//_____________________________________________________________________________
void RooTrace::active(Bool_t flag) 
{ 
  // If flag is true, memory tracing is activated

  _active = flag ; 
}


//_____________________________________________________________________________
void RooTrace::verbose(Bool_t flag) 
{ 
  // If flag is true, a message will be printed at each
  // object creation or deletion

  _verbose = flag ; 
}



//_____________________________________________________________________________
void RooTrace::create2(const TObject* obj) 
{
  // Back end function of create(), register creation of object 'obj' 
  
  _list.Add((RooAbsArg*)obj) ;
  if (_verbose) {
    cout << "RooTrace::create: object " << obj << " of type " << obj->ClassName() 
	 << " created " << endl ;
  }
}


  

//_____________________________________________________________________________
void RooTrace::destroy2(const TObject* obj) 
{
  // Back end function of destroy(), register deletion of object 'obj' 

  if (!_list.Remove((RooAbsArg*)obj)) {
  } else if (_verbose) {
    cout << "RooTrace::destroy: object " << obj << " of type " << obj->ClassName() 
	 << " destroyed [" << obj->GetTitle() << "]" << endl ;
  }
}



//_____________________________________________________________________________
void RooTrace::mark()
{
  // Put marker in object list, that allows to dump contents of list
  // relative to this marker

  _markList = _list ;
}



//_____________________________________________________________________________
void RooTrace::dump() 
{
  // Dump contents of object registry to stdout
  dump(cout,kFALSE) ;
}


//_____________________________________________________________________________
void RooTrace::dump(ostream& os, Bool_t sinceMarked) 
{
  // Dump contents of object register to stream 'os'. If sinceMarked is
  // true, only object created after the last call to mark() are shown.

  os << "List of RooFit objects allocated while trace active:" << endl ;


  Int_t i, nMarked(0) ;
  for(i=0 ; i<_list.GetSize() ; i++) {
    if (!sinceMarked || _markList.IndexOf(_list.At(i)) == -1) {
      os << hex << setw(10) << _list.At(i) << dec << " : " << setw(20) << _list.At(i)->ClassName() << setw(0) << " - " << _list.At(i)->GetName() << endl ;
    } else {
      nMarked++ ;
    }
  }
  if (sinceMarked) os << nMarked << " marked objects suppressed" << endl ;
}
