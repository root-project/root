/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTrace.cc,v 1.3 2001/08/09 01:02:15 verkerke Exp $
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


Bool_t RooTrace::_active(kFALSE) ;
Bool_t RooTrace::_verbose(kFALSE) ;
Bool_t RooTrace::_pad(kFALSE) ;
RooPadTable RooTrace::_rpt ;
char RooPad::_fil('A') ;

RooPadTable::RooPadTable() 
{
  // Clear initial values
  Int_t i ;
  for (i=0 ; i<size ; i++) {
    _padA[i]=0 ;
    _refA[i]=0 ;
  }
  _hwm = 0 ;
  _lfm = 0 ;
}


void RooPadTable::addPad(const TObject* ref, Bool_t doPad)
{
  //cout << "addPad: filling slot " << _lfm << endl ;
  _padA[_lfm] =  doPad ? new RooPad : 0 ;
  _refA[_lfm] = (TObject*) ref ;

  // Find next free block 
  _lfm++ ;
  while (_refA[_lfm]!=0 && _lfm<size) _lfm++ ;
  //cout << "addPad: increasing LFM to " << _lfm  << endl ; 

  // Update hwm is necessary
  if (_lfm>=_hwm) {
    //cout << "addPad: creasing HWM to LFM" << endl ;
    _hwm=_lfm ;
  }
  // Crude protection against overflows
  if (_lfm==size-1) assert(0) ;
}


Bool_t RooPadTable::removePad(const TObject* ref)
{
  Int_t i ;
  for (i=0 ; i<_hwm ; i++) {
    if (_refA[i]==ref) {

      // Delete and zero matching entry
      if (_padA[i]) delete _padA[i] ; 
      _padA[i]=0 ;
      _refA[i]=0 ;
      //cout << "removePad: clearing slot " << i << endl ;

      // Lower lfm if necessary 
      if (_lfm>i) {
	//cout << "removePad: lowering LFM to " << i << endl ;
	_lfm=i ;
      }
      return kFALSE;
    }
  }
  return kTRUE ;
}
  


void RooPadTable::checkPads() 
{
  static Int_t ncalls(0) ;
  Int_t i ;
  Int_t n(0) ;
  for(i=0 ; i<_hwm ; i++) {
    if (_padA[i]) {
      n++ ;
      if (_padA[i]->check()) {
	cout << "Above pad errors associated with reference object " << _refA[i] << endl ;
      }
    }
  }
  if (++ncalls%100==0) cout << "(" << ncalls << ")Currently " << n << " pads in memory" << endl ;
}




void RooTrace::create2(const TObject* obj) {
  if (_pad) _rpt.checkPads() ;

  _rpt.addPad(obj,_pad) ;
  if (_verbose) {
    cout << "RooTrace::create: object " << obj << " of type " << obj->ClassName() 
	 << " created " << endl ;
  }
}


  
void RooTrace::destroy2(const TObject* obj) {
  if (_pad) _rpt.checkPads() ;

  if (_rpt.removePad(obj)) {
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
  for(i=0 ; i<_rpt._hwm ; i++) {
    if (_rpt._refA[i]) {
      sprintf(buf,"%010x : ",(void*)_rpt._refA[i]) ;
      os << buf << setw(20) << _rpt._refA[i]->ClassName() << setw(0) << " - " << _rpt._refA[i]->GetName() << endl ;
    }
  }
}
