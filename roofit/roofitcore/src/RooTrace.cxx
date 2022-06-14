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

/**
\file RooTrace.cxx
\class RooTrace
\ingroup Roofitcore

Class RooTrace controls the memory tracing hooks in all RooFit
objects. When tracing is active, a table of live RooFit objects
is kept that can be queried at any time. In verbose mode, messages
are printed in addition at the construction and destruction of
each object.
**/

#include "RooTrace.h"
#include "RooAbsArg.h"
#include "Riostream.h"
#include "RooMsgService.h"

#include <iomanip>
#include "TClass.h"


using namespace std;

ClassImp(RooTrace);
;

RooTrace* RooTrace::_instance=0 ;


////////////////////////////////////////////////////////////////////////////////

RooTrace& RooTrace::instance()
{
  if (_instance==0) _instance = new RooTrace() ;
  return *_instance ;
}


////////////////////////////////////////////////////////////////////////////////

RooTrace::RooTrace() : _active(false), _verbose(false)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Register creation of object 'obj'

void RooTrace::create(const TObject* obj)
{
  RooTrace& instance = RooTrace::instance() ;
  if (instance._active) {
    instance.create3(obj) ;
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Register deletion of object 'obj'

void RooTrace::destroy(const TObject* obj)
{
  RooTrace& instance = RooTrace::instance() ;
  if (instance._active) {
    instance.destroy3(obj) ;
  }
}


////////////////////////////////////////////////////////////////////////////////

void RooTrace::createSpecial(const char* name, int size)
{
  RooTrace& instance = RooTrace::instance() ;
  if (instance._active) {
    instance.createSpecial3(name,size) ;
  }
}


////////////////////////////////////////////////////////////////////////////////

void RooTrace::destroySpecial(const char* name)
{
  RooTrace& instance = RooTrace::instance() ;
  if (instance._active) {
    instance.destroySpecial3(name) ;
  }
}


////////////////////////////////////////////////////////////////////////////////

void RooTrace::createSpecial3(const char* name, int size)
{
  _specialCount[name]++ ;
  _specialSize[name] = size ;
}


////////////////////////////////////////////////////////////////////////////////

void RooTrace::destroySpecial3(const char* name)
{
  _specialCount[name]-- ;
}



////////////////////////////////////////////////////////////////////////////////
/// If flag is true, memory tracing is activated

void RooTrace::active(bool flag)
{
  RooTrace::instance()._active = flag ;
}


////////////////////////////////////////////////////////////////////////////////
/// If flag is true, a message will be printed at each
/// object creation or deletion

void RooTrace::verbose(bool flag)
{
  RooTrace::instance()._verbose = flag ;
}





////////////////////////////////////////////////////////////////////////////////
/// Back end function of create(), register creation of object 'obj'

void RooTrace::create2(const TObject* obj)
{
  _list.Add((RooAbsArg*)obj) ;
  if (_verbose) {
    cout << "RooTrace::create: object " << obj << " of type " << obj->ClassName()
    << " created " << endl ;
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Back end function of destroy(), register deletion of object 'obj'

void RooTrace::destroy2(const TObject* obj)
{
  if (!_list.Remove((RooAbsArg*)obj)) {
  } else if (_verbose) {
    cout << "RooTrace::destroy: object " << obj << " of type " << obj->ClassName()
    << " destroyed [" << obj->GetTitle() << "]" << endl ;
  }
}



//_____________________________________________________________________________

void RooTrace::create3(const TObject* obj)
{
  // Back end function of create(), register creation of object 'obj'
  _objectCount[obj->IsA()]++ ;
}




////////////////////////////////////////////////////////////////////////////////
/// Back end function of destroy(), register deletion of object 'obj'

void RooTrace::destroy3(const TObject* obj)
{
  _objectCount[obj->IsA()]-- ;
}



////////////////////////////////////////////////////////////////////////////////
/// Put marker in object list, that allows to dump contents of list
/// relative to this marker

void RooTrace::mark()
{
  RooTrace::instance().mark3() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Put marker in object list, that allows to dump contents of list
/// relative to this marker

void RooTrace::mark3()
{
  _markList = _list ;
}



////////////////////////////////////////////////////////////////////////////////
/// Dump contents of object registry to stdout

void RooTrace::dump()
{
  RooTrace::instance().dump3(cout,false) ;
}


////////////////////////////////////////////////////////////////////////////////

void RooTrace::dump(ostream& os, bool sinceMarked)
{
  RooTrace::instance().dump3(os,sinceMarked) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Dump contents of object register to stream 'os'. If sinceMarked is
/// true, only object created after the last call to mark() are shown.

void RooTrace::dump3(ostream& os, bool sinceMarked)
{
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


////////////////////////////////////////////////////////////////////////////////

void RooTrace::printObjectCounts()
{
  RooTrace::instance().printObjectCounts3() ;
}

////////////////////////////////////////////////////////////////////////////////

void RooTrace::printObjectCounts3()
{
  double total(0) ;
  for (map<TClass*,int>::iterator iter = _objectCount.begin() ; iter != _objectCount.end() ; ++iter) {
    double tot= 1.0*(iter->first->Size()*iter->second)/(1024*1024) ;
    cout << " class " << iter->first->GetName() << " count = " << iter->second << " sizeof = " << iter->first->Size() << " total memory = " <<  Form("%5.2f",tot) << " Mb" << endl ;
    total+=tot ;
  }

  for (map<string,int>::iterator iter = _specialCount.begin() ; iter != _specialCount.end() ; ++iter) {
    int size = _specialSize[iter->first] ;
    double tot=1.0*(size*iter->second)/(1024*1024) ;
    cout << " speeial " << iter->first << " count = " << iter->second << " sizeof = " << size  << " total memory = " <<  Form("%5.2f",tot) << " Mb" << endl ;
    total+=tot ;
  }
  cout << "Grand total memory = " << Form("%5.2f",total) << " Mb" << endl ;

}


////////////////////////////////////////////////////////////////////////////////
/// Utility function to trigger zeroing of callgrind counters.
///
/// Note that this function does _not_ do anything, other than optionally printing this message
/// To trigger callgrind zero counter action, run callgrind with
/// argument '--zero-before=RooTrace::callgrind_zero()' (include single quotes in cmdline)

void RooTrace::callgrind_zero()
{
  ooccoutD((TObject*)0,Tracing) << "RooTrace::callgrind_zero()" << endl ;
}

////////////////////////////////////////////////////////////////////////////////
/// Utility function to trigger dumping of callgrind counters.
///
/// Note that this function does _not_ do anything, other than optionally printing this message
/// To trigger callgrind dumping action, run callgrind with
/// argument '--dump-before=RooTrace::callgrind_dump()' (include single quotes in cmdline)

void RooTrace::callgrind_dump()
{
  ooccoutD((TObject*)0,Tracing) << "RooTrace::callgrind_dump()" << endl ;
}
