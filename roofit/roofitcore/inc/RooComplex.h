/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooComplex.h,v 1.13 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_COMPLEX
#define ROO_COMPLEX

#if !defined(ROO_MATH) && !defined(ROO_COMPLEX_CXX) && !defined(__CINT__) && !defined(__CLING__) && \
    !defined(R__DICTIONARY_FILENAME)
#warning "RooComplex is deprecated, use std::complex instead!"
#endif

#include <math.h>
#include "Rtypes.h"
#include "Riosfwd.h"
#include <complex>

// This is a bare-bones complex class adapted from the CINT complex.h header,
// and introduced to support the complex error function in RooMath. The main
// changes with respect to the CINT header are to avoid defining global
// functions (at the cost of not supporting implicit casts on the first
// argument) and adding const declarations where appropriate.

class RooComplex {
public:

  inline RooComplex(std::complex<Double_t> c) : _re(c.real()), _im(c.imag()) { }

  inline RooComplex(Double_t a=0, Double_t b=0) : _re(a), _im(b) { warn(); }
  virtual ~RooComplex() { }
  inline RooComplex& operator=(const RooComplex& other) {
    warn();
    if (&other==this) return *this ;
    this->_re= other._re;
    this->_im= other._im;
    return(*this);
  }
  // unary operators
  inline RooComplex operator-() const {
    return RooComplex(-_re,-_im);
  }
  // binary operators
  inline RooComplex operator+(const RooComplex& other) const {
    return RooComplex(this->_re + other._re, this->_im + other._im);
  }
  inline RooComplex operator-(const RooComplex& other) const {
    return RooComplex(this->_re - other._re, this->_im - other._im);
  }
  inline RooComplex operator*(const RooComplex& other) const {
    return RooComplex(this->_re*other._re - this->_im*other._im,
		      this->_re*other._im + this->_im*other._re);
  }
  inline RooComplex operator/(const RooComplex& other) const {
    Double_t x(other.abs2());
    return RooComplex((this->_re*other._re + this->_im*other._im)/x,
		      (this->_im*other._re - this->_re*other._im)/x);
  }
  inline RooComplex operator*(const Double_t& other) const {
    return RooComplex(this->_re*other,this->_im*other);
  }


  inline Bool_t operator==(const RooComplex& other) const {
    return (_re==other._re && _im==other._im) ;
  }

  // unary functions
  inline Double_t re() const {
    return _re;
  }
  inline Double_t im() const {
    return _im;
  }
  inline Double_t abs() const {
    return ::sqrt(_re*_re + _im*_im);
  }
  inline Double_t abs2() const {
    return _re*_re + _im*_im;
  }
  inline RooComplex exp() const {
    Double_t mag(::exp(_re));
    return RooComplex(mag*cos(_im),mag*sin(_im));
  }
  inline RooComplex conj() const {
    return RooComplex(_re,-_im);
  }
  inline RooComplex sqrt() const {
    Double_t arg=atan2(_im,_re)*0.5;
    Double_t mag=::sqrt(::sqrt(_re*_re + _im*_im));
    return RooComplex(mag*cos(arg),mag*sin(arg));
  }
  // ouptput formatting
  void Print() const;
private:
  Double_t _re,_im;

  void warn() const;

  ClassDef(RooComplex,0) // a non-persistent bare-bones complex class
};

// output formatting
std::ostream& operator<<(std::ostream& os, const RooComplex& z);

#endif
