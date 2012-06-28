/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* Complex.h
*
*
**************************************************************************/

#ifndef COMPLEX_H
#define COMPLEX_H

#include <stdio.h>
#include <math.h>
#include <iostream>

class Complex {
 public:
  Complex(double a=0.0,double b=0.0) { re=a; im=b; }
  Complex(const Complex& x) { re=x.re; im=x.im; }
  //Complex& operator=(const Complex& x) { re=x.re; im=x.im; return *this;}
  double real(void) { return(re); }
  double imag(void) { return(im); }
  double abs(void) { return(sqrt(re*re+im*im)); }
  void disp(void) ;

  friend Complex operator +(Complex const & a,Complex const & b);
  friend Complex operator -(Complex const & a,Complex const & b);
  friend Complex operator *(Complex const & a,Complex const & b);
  friend Complex operator /(Complex const & a,Complex const & b);
  friend Complex exp(Complex& a);
  friend double fabs(Complex& a);
  friend double real(Complex& a);
  friend double imag(Complex& a);

 private:
  double re,im;
};

Complex operator +(Complex const & a,Complex const & b);
Complex operator -(Complex const & a,Complex const & b);
Complex operator *(Complex const & a,Complex const & b);
Complex operator /(Complex const & a,Complex const & b);
Complex exp(Complex& a);
double fabs(Complex& a);
double real(Complex& a);
double imag(Complex& a);

using namespace std;
std::ostream& operator<<(std::ostream& ios,Complex& a);

// Added for test
const int cf1(const int a);
const int& cf2(const int& a);
int const& cf3(int const& a);
int const & cf4(int const & a);
int const &cf5(int const &a);
const int & cf6(const int & a);
const int* cf7(const int* a);
const int *const cf8(const int *const a);
const int * const cf9(const int * const a);

#endif
