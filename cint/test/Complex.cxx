/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Complex.cxx

#include <math.h>
#include "Complex.h"

// メンバ関数による演算子多重定義 //////////////////////////////////
Complex& Complex::operator+=(Complex& a) {
  re += a.re;
  im += a.im;
  return(*this);
}

Complex& Complex::operator-=(Complex& a) {
  re -= a.re;
  im -= a.im;
  return(*this);
}

Complex& Complex::operator*=(Complex& a) {
  re = re*a.re-im*a.im;
  im = re*a.im+im*a.re;
  return(*this);
}

Complex& Complex::operator/=(Complex& a) {
  double x;
  x = a.re*a.re+a.im*a.im;
  re = (re*a.re+im*a.im)/x;
  im = (im*a.re-re*a.im)/x;
 return(*this);
}

// フレンド関数による演算子多重定義 ////////////////////////////////
bool operator ==(Complex& a,Complex& b)
{
  return( a.re==b.re && a.im==b.im );
}

Complex operator +(Complex& a,Complex& b)
{
  Complex c;
  c.re = a.re+b.re;
  c.im = a.im+b.im;
  return(c);
}

Complex operator -(Complex& a,Complex& b)
{
  Complex c;
  c.re = a.re-b.re;
  c.im = a.im-b.im;
  return(c);
}

Complex operator *(Complex& a,Complex& b)
{
  Complex c;
  c.re = a.re*b.re-a.im*b.im;
  c.im = a.re*b.im+a.im*b.re;
  return(c);
}

Complex operator /(Complex& a,Complex& b)
{
  Complex c;
  double x;
  x = b.re*b.re+b.im*b.im;
  c.re = (a.re*b.re+a.im*b.im)/x;
  c.im = (a.im*b.re-a.re*b.im)/x;
  return(c);
}

// 入出力ストリーム演算子の多重定義 //////////////////////////////
ostream& operator <<(ostream& ios,Complex& a)
{
  ios << "(" << a.re << "," << a.im << ")" ;
  return(ios);
}

// 算術関数の多重定義 ////////////////////////////////////////////
double abs(Complex &a)
{
  double result = sqrt(a.re*a.re+a.im*a.im);
  return(result);
}

Complex exp(Complex& a)
{
  Complex c;
  double mag = exp(a.re);
  c.re=mag*cos(a.im);
  c.im=mag*sin(a.im);
  return(c);
}
