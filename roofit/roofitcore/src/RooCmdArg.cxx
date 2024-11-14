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
\file RooCmdArg.cxx
\class RooCmdArg
\ingroup Roofitcore

Named container for two doubles, two integers
two object points and three string pointers that can be passed
as generic named arguments to a variety of RooFit end user
methods. To achieved the named syntax, RooCmdArg objects are
created using global helper functions defined in RooGlobalFunc.h
that create and fill these generic containers
**/



#include "RooCmdArg.h"
#include "Riostream.h"
#include "RooArgSet.h"

#include "RooFitImplHelpers.h"

#include <array>
#include <sstream>
#include <string>
#include <iostream>

ClassImp(RooCmdArg);

const RooCmdArg RooCmdArg::_none ;


////////////////////////////////////////////////////////////////////////////////
/// Return reference to null argument

const RooCmdArg& RooCmdArg::none()
{
  return _none ;
}


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooCmdArg::RooCmdArg() : TNamed("","")
{
  _o[0] = nullptr ;
  _o[1] = nullptr ;
  _i[0] = 0 ;
  _i[1] = 0 ;
  _d[0] = 0 ;
  _d[1] = 0 ;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with full specification of payload: two integers, two doubles,
/// three string poiners, two object pointers and one RooCmdArg pointer

RooCmdArg::RooCmdArg(const char* name, Int_t i1, Int_t i2, double d1, double d2,
           const char* s1, const char* s2, const TObject* o1, const TObject* o2,
           const RooCmdArg* ca, const char* s3, const RooArgSet* c1, const RooArgSet* c2) :
  TNamed(name,name)
{
  _i[0] = i1 ;
  _i[1] = i2 ;
  _d[0] = d1 ;
  _d[1] = d2 ;
  if (s1) _s[0] = s1 ;
  if (s2) _s[1] = s2 ;
  if (s3) _s[2] = s3 ;
  _o[0] = const_cast<TObject*>(o1);
  _o[1] = const_cast<TObject*>(o2);
  _c = nullptr ;

  if (c1||c2) _c = new RooArgSet[2] ;
  if (c1) _c[0].add(*c1) ;
  if (c2) _c[1].add(*c2) ;

  _procSubArgs = true ;
  _prefixSubArgs = true ;
  if (ca) {
    _argList.Add(new RooCmdArg(*ca)) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooCmdArg::RooCmdArg(const RooCmdArg& other) :
  TNamed(other)
{
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
    _c = nullptr ;
  }

  _procSubArgs = other._procSubArgs ;
  _prefixSubArgs = other._prefixSubArgs ;
  for (Int_t i=0 ; i<other._argList.GetSize() ; i++) {
    _argList.Add(new RooCmdArg(static_cast<RooCmdArg&>(*other._argList.At(i)))) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

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
    _argList.Add(new RooCmdArg(static_cast<RooCmdArg&>(*other._argList.At(i)))) ;
  }

  return *this ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooCmdArg::~RooCmdArg()
{
  _argList.Delete() ;
  if (_c) delete[] _c ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function to add nested RooCmdArg to payload of this RooCmdArg

void RooCmdArg::addArg(const RooCmdArg& arg)
{
  _argList.Add(new RooCmdArg(arg)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return RooArgSet stored in slot idx

const RooArgSet* RooCmdArg::getSet(Int_t idx) const {
    return _c ? &_c[idx] : nullptr ;
  }



////////////////////////////////////////////////////////////////////////////////

void RooCmdArg::setSet(Int_t idx,const RooArgSet& set)
{
  if (!_c) {
    _c = new RooArgSet[2] ;
  }
    _c[idx].removeAll() ;
    _c[idx].add(set) ;
}

std::string RooCmdArg::constructorCode() const
{
   std::array<bool, 13> needs;
   needs[0] = true;                  // name
   needs[1] = true;                  // i1
   needs[2] = _i[1] != 0;            // i2
   needs[3] = _d[0] != 0;            // d1
   needs[4] = _d[1] != 0;            // d2
   needs[5] = !_s[0].empty();        // s1
   needs[6] = !_s[1].empty();        // s2
   needs[7] = _o[0];                 // o1
   needs[8] = _o[1];                 // o2
   needs[9] = !_argList.empty();     // ca
   needs[10] = !_s[2].empty();       // s3
   needs[11] = _c;                   // c1
   needs[12] = _c && !_c[1].empty(); // c2

   // figure out until which point we actually need to pass constructor
   // arguments
   bool b = false;
   for (int i = needs.size() - 1; i >= 0; --i) {
      b |= needs[i];
      needs[i] = b;
   }

   std::stringstream ss;

   // The first two arguments always need to be passed
   ss << "RooCmdArg(\"" << GetName() << "\", " << _i[0];

   if (needs[2])
      ss << ", " << _i[1];
   if (needs[3])
      ss << ", " << _d[0];
   if (needs[4])
      ss << ", " << _d[1];
   if (needs[5])
      ss << ", " << (!_s[0].empty() ? "\"" + _s[0] + "\"" : "\"\"");
   if (needs[6])
      ss << ", " << (!_s[1].empty() ? "\"" + _s[1] + "\"" : "\"\"");
   if (needs[7])
      ss << ", " << (_o[0] ? "\"" + std::string(_o[0]->GetName()) + "\"" : "0");
   if (needs[8])
      ss << ", " << (_o[1] ? "\"" + std::string(_o[1]->GetName()) + "\"" : "0");
   if (needs[9]) {
      ss << ", ";
      if (!_argList.empty()) {
         ss << "{\n";
         for (std::size_t i = 0; i < _argList.size(); ++i) {
            if (auto *cmdArg = dynamic_cast<RooCmdArg *>(_argList.At(i))) {
               ss << cmdArg->constructorCode() << "\n";
            }
         }
         ss << "}\n";
      } else {
         ss << 0;
      }
   }
   if (needs[10])
      ss << ", " << (!_s[2].empty() ? "\"" + _s[2] + "\"" : "\"\"");
   if (needs[11])
      ss << ", RooArgSet(" << RooHelpers::getColonSeparatedNameString(_c[0], ',') << ")";
   if (needs[12])
      ss << ", RooArgSet(" << RooHelpers::getColonSeparatedNameString(_c[1], ',') << ")";
   ss << ")";

   return ss.str();
}

////////////////////////////////////////////////////////////////////////////////
// Print contents
void RooCmdArg::Print(const char *opts) const
{
   TString o{opts};
   if (o.Contains("v")) {
      std::cout << constructorCode() << std::endl;
      return;
   }

   std::cout << GetName() << ":\ndoubles\t" << _d[0] << " " << _d[1] << "\nints\t" << _i[0] << " " << _i[1]
             << "\nstrings\t" << _s[0] << " " << _s[1] << " " << _s[2] << "\nobjects\t" << _o[0] << " " << _o[1]
             << std::endl;
}
