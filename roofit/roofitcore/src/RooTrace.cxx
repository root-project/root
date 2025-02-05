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

Controls the memory tracing hooks in all RooFit
objects. When tracing is active, a table of live RooFit objects
is kept that can be queried at any time. In verbose mode, messages
are printed in addition at the construction and destruction of
each object.

Usage example:
\code{.cpp}
void exampleRooTrace()
{
   using namespace RooFit;

   // Activate RooFit memory tracing
   RooTrace::active(true);
 
   // Construct gauss(x,m,s)
   RooRealVar x("x", "x", -10, 10);
   RooRealVar m("m", "m", 0, -10, 10);
   RooRealVar s("s", "s", 1, -10, 10);
   RooGaussian gauss("g", "g", x, m, s);
 
   // Show dump of all RooFit object in memory
   RooTrace::dump();
 
   // Activate verbose mode
   RooTrace::verbose(true);
 
   // Construct poly(x,p0)
   RooRealVar p0("p0", "p0", 0.01, 0., 1.);
   RooPolynomial poly("p", "p", x, p0);
 
   // Put marker in trace list for future reference
   RooTrace::mark();
 
   // Construct model = f*gauss(x) + (1-f)*poly(x)
   RooRealVar f("f", "f", 0.5, 0., 1.);
   RooAddPdf model("model", "model", RooArgSet(gauss, poly), f);
 
   // Show object added to memory since marker
   RooTrace::printObjectCounts();
 
   // Since verbose mode is still on, you will see messages
   // pertaining to destructor calls of all RooFit objects
   // made in this macro
   //
   // A call to RooTrace::dump() at the end of this macro
   // should show that there a no RooFit object left in memory
}
\endcode

\note In the ROOT releases, the RooTrace is disabled at compile time and the
example above will not print any objects. If you are an advanced developer who
wants to use the RooTrace, you need to recompile ROOT after changing the
`TRACE_CREATE` and `TRACE_DESTROY` macros in RooTrace.h to call the RooTrace
functions:

\code{.cpp}
#define TRACE_CREATE RooTrace::create(this);
#define TRACE_DESTROY RooTrace::destroy(this);
\endcode

However, as ROOT is not build with this by default, the RooTrace is not tested
and there is no guarantee that this works.
**/

#include "RooTrace.h"
#include "RooAbsArg.h"
#include "Riostream.h"
#include "RooMsgService.h"

#include <iomanip>
#include "TClass.h"


using std::ostream, std::setw, std::hex, std::dec, std::map, std::string;


RooTrace* RooTrace::_instance=nullptr ;


////////////////////////////////////////////////////////////////////////////////

RooTrace& RooTrace::instance()
{
  if (_instance==nullptr) _instance = new RooTrace() ;
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
  _list.Add(const_cast<RooAbsArg *>(static_cast<RooAbsArg const*>(obj)));
  if (_verbose) {
    std::cout << "RooTrace::create: object " << obj << " of type " << obj->ClassName()
    << " created " << std::endl ;
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Back end function of destroy(), register deletion of object 'obj'

void RooTrace::destroy2(const TObject* obj)
{
  if (!_list.Remove(const_cast<RooAbsArg *>(static_cast<RooAbsArg const*>(obj)))) {
  } else if (_verbose) {
    std::cout << "RooTrace::destroy: object " << obj << " of type " << obj->ClassName()
    << " destroyed [" << obj->GetTitle() << "]" << std::endl ;
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
  RooTrace::instance().dump3(std::cout,false) ;
}


////////////////////////////////////////////////////////////////////////////////

void RooTrace::dump(std::ostream& os, bool sinceMarked)
{
  RooTrace::instance().dump3(os,sinceMarked) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Dump contents of object register to stream 'os'. If sinceMarked is
/// true, only object created after the last call to mark() are shown.

void RooTrace::dump3(std::ostream& os, bool sinceMarked)
{
  os << "List of RooFit objects allocated while trace active:" << std::endl ;

  Int_t i;
  Int_t nMarked(0);
  for(i=0 ; i<_list.GetSize() ; i++) {
    if (!sinceMarked || _markList.IndexOf(_list.At(i)) == -1) {
      os << hex << setw(10) << _list.At(i) << dec << " : " << setw(20) << _list.At(i)->ClassName() << setw(0) << " - " << _list.At(i)->GetName() << std::endl ;
    } else {
      nMarked++ ;
    }
  }
  if (sinceMarked) os << nMarked << " marked objects suppressed" << std::endl ;
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
    std::cout << " class " << iter->first->GetName() << " count = " << iter->second << " sizeof = " << iter->first->Size() << " total memory = " <<  Form("%5.2f",tot) << " Mb" << std::endl ;
    total+=tot ;
  }

  for (map<string,int>::iterator iter = _specialCount.begin() ; iter != _specialCount.end() ; ++iter) {
    int size = _specialSize[iter->first] ;
    double tot=1.0*(size*iter->second)/(1024*1024) ;
    std::cout << " speeial " << iter->first << " count = " << iter->second << " sizeof = " << size  << " total memory = " <<  Form("%5.2f",tot) << " Mb" << std::endl ;
    total+=tot ;
  }
  std::cout << "Grand total memory = " << Form("%5.2f",total) << " Mb" << std::endl ;

}


////////////////////////////////////////////////////////////////////////////////
/// Utility function to trigger zeroing of callgrind counters.
///
/// Note that this function does _not_ do anything, other than optionally printing this message
/// To trigger callgrind zero counter action, run callgrind with
/// argument '--zero-before=RooTrace::callgrind_zero()' (include single quotes in cmdline)

void RooTrace::callgrind_zero()
{
  ooccoutD((TObject*)nullptr,Tracing) << "RooTrace::callgrind_zero()" << std::endl ;
}

////////////////////////////////////////////////////////////////////////////////
/// Utility function to trigger dumping of callgrind counters.
///
/// Note that this function does _not_ do anything, other than optionally printing this message
/// To trigger callgrind dumping action, run callgrind with
/// argument '--dump-before=RooTrace::callgrind_dump()' (include single quotes in cmdline)

void RooTrace::callgrind_dump()
{
  ooccoutD((TObject*)nullptr,Tracing) << "RooTrace::callgrind_dump()" << std::endl ;
}
