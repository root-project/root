/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTrace.cc,v 1.1 2001/08/02 21:39:13 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooTrace.hh"

#include <iomanip.h>

ClassImp(RooTrace)
;


RooTraceObj* RooTrace::_traceList(0) ;
Bool_t RooTrace::_active(kFALSE) ;
Bool_t RooTrace::_verbose(kFALSE) ;
Bool_t RooTrace::_pad(kFALSE) ;

void RooTrace::checkPad() 
{
  RooTraceObj* link(_traceList) ;
  while(link) {      
    link->checkPad() ;
    link = link->next() ;
  }
}


void RTcheckPad() {
  RooTrace::checkPad() ;
}


void RooTrace::create2(const TObject* obj) {
  if (_pad) checkPad() ;

  addToList(obj) ;
  if (_verbose) {
    cout << "RooTrace::create: object " << obj << " of type " << obj->ClassName() 
	 << " created " << endl ;
  }
}


  
void RooTrace::destroy2(const TObject* obj) {

  if (_pad) checkPad() ;
  
  if (!removeFromList(obj)) {
    cout << "RooTrace::destroy: object " << obj << " of type " << obj->ClassName() 
	 << " already deleted, or created before trace activation[" << obj->GetTitle() << "]" << endl ;
    assert(0) ;
  } else if (_verbose) {
    cout << "RooTrace::destroy: object " << obj << " of type " << obj->ClassName() 
	 << " destroyed [" << obj->GetTitle() << "]" << endl ;
  }
}


void RooTrace::dump(ostream& os) {
  os << "List of TObjects objects in memory while trace active:" << endl ;
  char buf[100] ;
  Int_t i ;
  RooTraceObj* link(_traceList) ;
  while(link) {
    sprintf(buf,"%010x : ",(void*)link->obj()) ;
    os << buf << setw(20) << link->obj()->ClassName() << setw(0) << " - " << link->obj()->GetName() << endl ;
    link = link->next() ;
  }
}
