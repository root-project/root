/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "RooFitCore/RooCmdArg.hh"
#include <iostream.h>

ClassImp(RooCmdArg)
  ;

RooCmdArg::RooCmdArg() : TNamed("","")
{
}

RooCmdArg::RooCmdArg(const char* name, Int_t i1, Int_t i2, Double_t d1, Double_t d2, 
		     const char* s1, const char* s2, const TObject* o1, const TObject* o2) :
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
}


RooCmdArg::~RooCmdArg()
{
}
