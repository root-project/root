/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooCmdArg.cxx,v 1.11 2007/05/11 09:11:58 verkerke Exp $
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

#include "RooFit.h"

#include "RooCmdArg.h"
#include "RooCmdArg.h"
#include "Riostream.h"
#include <string>

ClassImp(RooCmdArg)
  ;

const RooCmdArg RooCmdArg::none ;

RooCmdArg::RooCmdArg() : TNamed("","")
{
  _procSubArgs = kFALSE ;
}

RooCmdArg::RooCmdArg(const char* name, Int_t i1, Int_t i2, Double_t d1, Double_t d2, 
		     const char* s1, const char* s2, const TObject* o1, const TObject* o2, const RooCmdArg* ca) :
  TNamed(name,name)
{
  _i[0] = i1 ;
  _i[1] = i2 ;
  _d[0] = d1 ;
  _d[1] = d2 ;
  _s[0] = s1 ;
  _s[1] = s2 ;
  _o[0] = (TObject*) o1 ;
  _o[1] = (TObject*) o2 ;
  _procSubArgs = kTRUE ;
  if (ca) {
    _argList.Add(new RooCmdArg(*ca)) ;
  }
}


RooCmdArg::RooCmdArg(const RooCmdArg& other) :
  TNamed(other)
{
  _i[0] = other._i[0] ;
  _i[1] = other._i[1] ;
  _d[0] = other._d[0] ;
  _d[1] = other._d[1] ;
  _s[0] = other._s[0] ;
  _s[1] = other._s[1] ;
  _o[0] = other._o[0] ;
  _o[1] = other._o[1] ;
  _procSubArgs = other._procSubArgs ;
  for (Int_t i=0 ; i<other._argList.GetSize() ; i++) {
    _argList.Add(new RooCmdArg((RooCmdArg&)*other._argList.At(i))) ;
  }
}

RooCmdArg& RooCmdArg::operator=(const RooCmdArg& other) 
{
  if (&other==this) return *this ;

  SetName(other.GetName()) ;
  SetTitle(other.GetTitle()) ;

  _i[0] = other._i[0] ;
  _i[1] = other._i[1] ;
  _d[0] = other._d[0] ;
  _d[1] = other._d[1] ;
  _s[0] = other._s[0] ;
  _s[1] = other._s[1] ;
  _o[0] = other._o[0] ;
  _o[1] = other._o[1] ;

  _procSubArgs = other._procSubArgs ;

  for (Int_t i=0 ; i<other._argList.GetSize() ; i++) {
    _argList.Add(new RooCmdArg((RooCmdArg&)*other._argList.At(i))) ;
  }

  return *this ;
}


RooCmdArg::~RooCmdArg()
{
  _argList.Delete() ;
}


void RooCmdArg::addArg(const RooCmdArg& arg) 
{
  _argList.Add(new RooCmdArg(arg)) ;
}
