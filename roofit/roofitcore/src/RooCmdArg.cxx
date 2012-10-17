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
// RooCmdArg is a named container for two doubles, two integers
// two object points and three string pointers that can be passed
// as generic named arguments to a variety of RooFit end user
// methods. To achieved the named syntax, RooCmdArg objects are
// created using global helper functions defined in RooGlobalFunc.h
// that create and fill these generic containers
// END_HTML
//

#include "RooFit.h"

#include "RooCmdArg.h"
#include "RooCmdArg.h"
#include "Riostream.h"
#include "RooArgSet.h"
#include <string>

ClassImp(RooCmdArg)
  ;

const RooCmdArg RooCmdArg::_none ;


//_____________________________________________________________________________
const RooCmdArg& RooCmdArg::none() 
{
  // Return reference to null argument
  return _none ;
}


//_____________________________________________________________________________
RooCmdArg::RooCmdArg() : TNamed("","")
{
  // Default constructor
  _procSubArgs = kFALSE ;
  _prefixSubArgs = kTRUE ;
  _c = 0 ;
  _o[0] = 0 ;
  _o[1] = 0 ; 
  _i[0] = 0 ;
  _i[1] = 0 ;
  _d[0] = 0 ;
  _d[1] = 0 ;
}


//_____________________________________________________________________________
RooCmdArg::RooCmdArg(const char* name, Int_t i1, Int_t i2, Double_t d1, Double_t d2, 
		     const char* s1, const char* s2, const TObject* o1, const TObject* o2, 
		     const RooCmdArg* ca, const char* s3, const RooArgSet* c1, const RooArgSet* c2) :
  TNamed(name,name)
{
  // Constructor with full specification of payload: two integers, two doubles,
  // three string poiners, two object pointers and one RooCmdArg pointer

  _i[0] = i1 ;
  _i[1] = i2 ;
  _d[0] = d1 ;
  _d[1] = d2 ;
  if (s1) _s[0] = s1 ;
  if (s2) _s[1] = s2 ;
  if (s3) _s[2] = s3 ;
  _o[0] = (TObject*) o1 ;
  _o[1] = (TObject*) o2 ;
  _c = 0 ;

  if (c1||c2) _c = new RooArgSet[2] ;
  if (c1) _c[0].add(*c1) ;
  if (c2) _c[1].add(*c2) ;

  _procSubArgs = kTRUE ;
  _prefixSubArgs = kTRUE ;
  if (ca) {
    _argList.Add(new RooCmdArg(*ca)) ;
  }
}



//_____________________________________________________________________________
RooCmdArg::RooCmdArg(const RooCmdArg& other) :
  TNamed(other)
{
  // Copy constructor

  _i[0] = other._i[0] ;
  _i[1] = other._i[1] ;
  _d[0] = other._d[0] ;
  _d[1] = other._d[1] ;
  _s[0] = other._s[0] ;
  _s[1] = other._s[1] ;
  _s[2] = other._s[2] ;
  _o[0] = other._o[0] ;
  _o[1] = other._o[1] ;
  if (other._c) {
    _c = new RooArgSet[2] ;
    _c[0].add(other._c[0]) ;
    _c[1].add(other._c[1]) ;
  } else {
    _c = 0 ;
  }
  
  _procSubArgs = other._procSubArgs ;
  _prefixSubArgs = other._prefixSubArgs ;
  for (Int_t i=0 ; i<other._argList.GetSize() ; i++) {
    _argList.Add(new RooCmdArg((RooCmdArg&)*other._argList.At(i))) ;
  }
}


//_____________________________________________________________________________
RooCmdArg& RooCmdArg::operator=(const RooCmdArg& other) 
{
  // Assignment operator

  if (&other==this) return *this ;

  SetName(other.GetName()) ;
  SetTitle(other.GetTitle()) ;

  _i[0] = other._i[0] ;
  _i[1] = other._i[1] ;
  _d[0] = other._d[0] ;
  _d[1] = other._d[1] ;
  _s[0] = other._s[0] ;
  _s[1] = other._s[1] ;
  _s[2] = other._s[2] ;
  _o[0] = other._o[0] ;
  _o[1] = other._o[1] ;
  if (!_c) _c = new RooArgSet[2] ;
  if (other._c) {
    _c[0].removeAll() ; _c[0].add(other._c[0]) ;
    _c[1].removeAll() ; _c[1].add(other._c[1]) ;
  }
  
  _procSubArgs = other._procSubArgs ;
  _prefixSubArgs = other._prefixSubArgs ;

  for (Int_t i=0 ; i<other._argList.GetSize() ; i++) {
    _argList.Add(new RooCmdArg((RooCmdArg&)*other._argList.At(i))) ;
  }

  return *this ;
}



//_____________________________________________________________________________
RooCmdArg::~RooCmdArg()
{
  // Destructor
  _argList.Delete() ;
  if (_c) delete[] _c ;
}



//_____________________________________________________________________________
void RooCmdArg::addArg(const RooCmdArg& arg) 
{
  // Utility function to add nested RooCmdArg to payload of this RooCmdArg

  _argList.Add(new RooCmdArg(arg)) ;
}



//_____________________________________________________________________________
const RooArgSet* RooCmdArg::getSet(Int_t idx) const {
  // Return RooArgSet stored in slot idx
    return _c ? &_c[idx] : 0 ;
  }



//_____________________________________________________________________________
void RooCmdArg::setSet(Int_t idx,const RooArgSet& set) 
{
  if (!_c) {
    _c = new RooArgSet[2] ;
  }
    _c[idx].removeAll() ;
    _c[idx].add(set) ;
}
