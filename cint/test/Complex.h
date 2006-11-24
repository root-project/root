/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Complex.h

#ifndef COMPLEX_H
#define COMPLEX_H

#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

// 複素数クラス //////////////////////////////////////////////////
class Complex {
 public:
  // 初期化
  Complex(double a=0.0,double b=0.0) { re=a; im=b; }

  // 情報取得
  double real() const { return(re); }
  double imag() const { return(im); }

  // メンバ関数による演算子多重定義
  Complex& operator+=(Complex& a);
  Complex& operator-=(Complex& a);
  Complex& operator*=(Complex& a);
  Complex& operator/=(Complex& a);

  // フレンド関数による演算子多重定義
  friend bool operator==(Complex& a,Complex& b);
  friend bool operator!=(Complex& a,Complex& b){return(!(a==b));}
  friend Complex operator +(Complex& a,Complex& b);
  friend Complex operator -(Complex& a,Complex& b);
  friend Complex operator *(Complex& a,Complex& b);
  friend Complex operator /(Complex& a,Complex& b);
  friend ostream& operator <<(ostream& ios,Complex& a);

  // 算術関数の多重定義
  friend double abs(Complex& a);
  friend Complex exp(Complex& a);

 private:
  double re,im;
};

#endif
