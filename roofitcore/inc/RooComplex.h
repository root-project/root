/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooComplex.rdl,v 1.1 2001/06/23 01:20:33 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jun-2000 DK Created initial version
 *   18-Jun-2001 WV Imported from RooFitTools
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_COMPLEX
#define ROO_COMPLEX

#include <math.h>
#include "Rtypes.h"
#include <iostream.h>

// This is a bare-bones complex class adapted from the CINT complex.h header,
// and introduced to support the complex error function in RooMath. The main
// changes with respect to the CINT header are to avoid defining global
// functions (at the cost of not supporting implicit casts on the first
// argument) and adding const declarations where appropriate.

class RooComplex {
public:
  inline RooComplex(Double_t a=0, Double_t b=0) : _re(a), _im(b) { }
  inline RooComplex& operator=(const RooComplex& other) {
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
  // unary functions
  inline Double_t re() const {
    return _re;
  }
  inline Double_t im() const {
    return _im;
  }
  inline Double_t abs() const {
    return sqrt(_re*_re + _im*_im);
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
  // ouptput formatting
  void Print() const;
private:
  Double_t _re,_im;
  ClassDef(RooComplex,0) // a non-persistent bare-bones complex class
};

// output formatting
ostream& operator<<(ostream& os, const RooComplex& z);

#endif
