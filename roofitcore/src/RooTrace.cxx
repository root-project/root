/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTrace.cc,v 1.9 2002/02/19 00:13:22 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --

#include "RooFitCore/RooTrace.hh"

#include <iomanip.h>

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
    assert(0) ;
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


  char buf[100] ;
  Int_t i, nMarked(0) ;
  for(i=0 ; i<_list.GetSize() ; i++) {
    if (!sinceMarked || _markList.IndexOf(_list.At(i)) == -1) {
      sprintf(buf,"%010x : ",(void*)_list.At(i)) ;
      os << buf << setw(20) << _list.At(i)->ClassName() << setw(0) << " - " << _list.At(i)->GetName() << endl ;
    } else {
      nMarked++ ;
    }
  }
  if (sinceMarked) os << nMarked << " marked objects suppressed" << endl ;
}
