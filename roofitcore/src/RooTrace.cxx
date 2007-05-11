/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTrace.cc,v 1.24 2005/06/22 18:02:33 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --

#include "RooFit.h"

#include "RooTrace.h"
#include "RooTrace.h"
#include "RooAbsArg.h"

#include <iomanip>

ClassImp(RooTrace)
;


Bool_t RooTrace::_active(kFALSE) ;
Bool_t RooTrace::_verbose(kFALSE) ;
RooLinkedList RooTrace::_list ;
RooLinkedList RooTrace::_markList ;

void RooTrace::create2(const TObject* obj) {
  
  _list.Add((RooAbsArg*)obj) ;
  if (_verbose) {
    cout << "RooTrace::create: object " << obj << " of type " << obj->ClassName() 
	 << " created " << endl ;
  }
}


  
void RooTrace::destroy2(const TObject* obj) {

  if (!_list.Remove((RooAbsArg*)obj)) {
    cout << "RooTrace::destroy: object " << obj << " of type " << obj->ClassName() 
	 << " already deleted, or created before trace activation[" << obj->GetTitle() << "]" << endl ;
  } else if (_verbose) {
    cout << "RooTrace::destroy: object " << obj << " of type " << obj->ClassName() 
	 << " destroyed [" << obj->GetTitle() << "]" << endl ;
  }
}


void RooTrace::mark()
{
  _markList = _list ;
}


void RooTrace::dump(ostream& os, Bool_t sinceMarked) {
  os << "List of RooFit objects allocated while trace active:" << endl ;


  Int_t i, nMarked(0) ;
  for(i=0 ; i<_list.GetSize() ; i++) {
    if (!sinceMarked || _markList.IndexOf(_list.At(i)) == -1) {
      os << hex << setw(10) << _list.At(i) << " : " << setw(20) << _list.At(i)->ClassName() << setw(0) << " - " << _list.At(i)->GetName() << endl ;
    } else {
      nMarked++ ;
    }
  }
  if (sinceMarked) os << nMarked << " marked objects suppressed" << endl ;
}
