/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   Nikhef & VU, Gerhard.Raven@nikhef.nl
 *                                                                           *
 * Copyright (c) 2010, Nikhef & VU. All rights reserved.
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"
#include <math.h>
#include <string>

#include "RooLegendre.h"
#include "RooAbsReal.h"
#include "Math/SpecFunc.h"
#include "TMath.h"

ClassImp(RooLegendre)
;


//_____________________________________________________________________________
namespace {
    inline double a(int p, int l, int m) {
        double r = TMath::Factorial(l+m)/TMath::Factorial(m+p)/TMath::Factorial(p)/TMath::Factorial(l-m-2*p);
        r /= pow(2.,m+2*p);
        return p%2==0 ? r : -r ;
    }
}

//_____________________________________________________________________________
RooLegendre::RooLegendre() :
  _l1(1),_m1(1),_l2(0),_m2(0)
{
}

//_____________________________________________________________________________
RooLegendre::RooLegendre(const char* name, const char* title, RooAbsReal& ctheta, int l, int m) 
 : RooAbsReal(name, title)
 , _ctheta("ctheta", "ctheta", this, ctheta)
 , _l1(l),_m1(m),_l2(0),_m2(0)
{
  //TODO: for now, we assume that ctheta has a range [-1,1]
  // should map the ctheta range onto this interval, and adjust integrals...

  //TODO: we assume m>=0
  //      should map m<0 back to m>=0...
}

//_____________________________________________________________________________
RooLegendre::RooLegendre(const char* name, const char* title, RooAbsReal& ctheta, int l1, int m1, int l2, int m2) 
 : RooAbsReal(name, title)
 , _ctheta("ctheta", "ctheta", this, ctheta)
 , _l1(l1),_m1(m1),_l2(l2),_m2(m2)
{
}

//_____________________________________________________________________________
RooLegendre::RooLegendre(const RooLegendre& other, const char* name) 
    : RooAbsReal(other, name)
    , _ctheta("ctheta", this, other._ctheta)
    , _l1(other._l1), _m1(other._m1)
    , _l2(other._l2), _m2(other._m2)
{
}

//_____________________________________________________________________________
Double_t RooLegendre::evaluate() const 
{
  // TODO: check that 0<=m_i<=l_i; on the other hand, assoc_legendre already does that ;-)
  // Note: P_0^0 = 1, so P_l^m = P_l^m P_0^0
#ifdef R__HAS_MATHMORE  
  double r = 1;
  if (_l1!=0||_m1!=0) r *= ROOT::Math::assoc_legendre(_l1,_m1,_ctheta);
  if (_l2!=0||_m2!=0) r *= ROOT::Math::assoc_legendre(_l2,_m2,_ctheta);
  if ((_m1+_m2)%2==1) r = -r;
  return r;
#else
  throw std::string("RooLegendre: ERROR: This class require installation of the MathMore library") ;
  return 0 ;
#endif
}

//_____________________________________________________________________________
namespace {
    bool fullRange(const RooRealProxy& x ,const char* range) 
    { return range==0 || strlen(range)==0 
          || ( x.min(range) == x.min() && x.max(range) == x.max() ) ; 
    }
}
Int_t RooLegendre::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const 
{
  // don't support indefinite integrals...
  if (fullRange(_ctheta,rangeName) && matchArgs(allVars, analVars, _ctheta)) return 1;
  return 0;
}

//_____________________________________________________________________________
Double_t RooLegendre::analyticalIntegral(Int_t code, const char* ) const 
{
  // this was verified to match mathematica for 
  // l1 in [0,2], m1 in [0,l1], l2 in [l1,4], m2 in [0,l2]
  assert(code==1) ;
  if ( _m1==_m2 )                 return ( _l1 == _l2) ?  TMath::Factorial(_l1+_m2)/TMath::Factorial(_l1-_m1)*double(2)/(2*_l1+1) : 0.;
  if ( (_l1+_l2-_m1-_m2)%2 != 0 ) return 0; // these combinations are odd under x -> -x

  // from B.R. Wong, "On the overlap integral of associated Legendre Polynomials" 1998 J. Phys. A: Math. Gen. 31 1101
  // TODO: update to the result of 
  //       H. A. Mavromatis
  //       "A single-sum expression for the overlap integral of two associated Legendre polynomials"
  //       1999 J. Phys. A: Math. Gen. 32 2601
  //       http://iopscience.iop.org/0305-4470/32/13/011/pdf/0305-4470_32_13_011.pdf
  //       For that we need Wigner 3-j, which Lorenzo has added for Root 5.28... (note: check Condon-Shortly convention in this paper!)
  double r=0;
  for (int p1=0; 2*p1 <= _l1-_m1 ;++p1) {
    double a1 = a(p1,_l1,_m1);
    for (int p2=0; 2*p2 <= _l2-_m2 ; ++p2) {
       double a2 = a(p2,_l2,_m2);
       r+= a1*a2*TMath::Gamma( double(_l1+_l2-_m1-_m2-2*p1-2*p2+1)/2 )*TMath::Gamma( double(_m1+_m2+2*p1+2*p2+2)/2 );
    }
  }
  r /= TMath::Gamma( double(_l1+_l2+3)/2 );

  if ((_m1+_m2)%2==1) r = -r;
  return r;
}

Int_t RooLegendre::getMaxVal( const RooArgSet& /*vars*/) const {
    if (_m1==0&&_m2==0) return 1;
    // does anyone know the analytical expression for the  max values in case m!=0??
    if (_l1<3&&_l2<3) return 1;
    return 0;
}

namespace {
    inline double maxSingle(int i, int j) {
        assert(j<=i);
        //   x0 : 1 (ordinary Legendre)
        if (j==0) return 1;
        assert(i<3);
        //   11: 1
        if (i<2) return 1;
        //   21: 3   22: 3
        static const double m2[3] = { 3,3 };
        return m2[j-1];
    }
}
Double_t RooLegendre::maxVal( Int_t /*code*/) const {
    return maxSingle(_l1,_m1)*maxSingle(_l2,_m2);
}
