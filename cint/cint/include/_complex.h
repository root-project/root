/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* complex
*
*  template version of complex
**************************************************************************/

#ifndef G__COMPLEX
#define G__COMPLEX

#include <iostream.h>
#include <math.h>

template<class scalar>
class complex {
 public:
   complex(scalar a=0,scalar b=0) { re=a; im=b; }
   complex(const complex& a) {re=a.re; im=a.im;}
   complex& operator=(const complex& a) { 
    this->re=a.re;
    this->im=a.im; 
    return(*this);
   }
   const scalar real() const { return re; }
   const scalar imag() const { return im; }
   scalar& real() { return re; }
   scalar& imag() { return im; }
 protected:
  scalar re,im;
};

template<class scalar>
complex<scalar>
operator +(complex<scalar>& a,complex<scalar>& b)
{
  return complex<scalar>(a.real()+b.real(), a.imag()+b.imag());
}
 
template<class scalar>
complex<scalar>
operator -(complex<scalar>& a,complex<scalar>& b)
{
  return complex<scalar>(a.real()-b.real(), a.imag()-b.imag());
}

template<class scalar>
complex<scalar>
operator *(complex<scalar>& a,complex<scalar>& b)
{
   return complex<scalar>(a.real()*b.real()-a.imag()*b.imag(), a.real()*b.imag()+a.imag()*b.real());
}

template<class scalar>
complex<scalar> operator /(complex<scalar>& a,complex<scalar>& b)
{
  scalar x;
  x = b.real()*b.real()+b.imag()*b.imag();
  return complex<scalar>((a.real()*b.real()+a.imag()*b.imag())/x,
                         (a.imag()*b.real()-a.real()*b.imag())/x);
}


//**********************************************************************

#ifndef __MAKECINT__
template<class scalar>
complex<scalar>
exp(const complex<scalar>& a)
{
  scalar mag;
  mag = exp(a.real());
  return complex<scalar>(mag*cos(a.imag()), mag*sin(a.imag()));
}
#endif

template<class scalar>
scalar
abs(const complex<scalar>& a)
{
  return sqrt(a.real()*a.real()+a.imag()*a.imag());
}

template <class scalar>
scalar 
re(complex<scalar>& a)
{
   return a.real();
}

template <class scalar>
scalar 
im(complex<scalar>& a)
{
   return a.imag();
}


template<class scalar>
scalar&
real(complex<scalar> &c) { return c.real(); }

template<class scalar>
scalar&
real(const complex<scalar> &c) { return c.real(); }

template<class scalar>
const scalar
imag(complex<scalar> &c) { return c.imag(); }

template<class scalar>
const scalar
imag(const complex<scalar> &c) { return c.imag(); }

template<class scalar>
scalar
abs(const complex<scalar> &c)
{
   return sqrt(c.real()*c.real()+c.imag()*c.imag());
}

template<class scalar>
scalar
arg(const complex<scalar> &c)
{ 
   return atan2(c.imag(), c.real());
}

template<class scalar>
scalar
norm(const complex<scalar> &c)
{
   return c.real()*c.real()+c.imag()*c.imag();
}

template<class scalar>
complex<scalar>
polar(scalar abs, scalar arg)
{
   return complex<scalar>(abs * cos(arg), abs * sin(arg));
}

template<class scalar>
complex<scalar>
conj(const complex<scalar> &c)
{
   return complex<scalar>(c.real(), -c.imag());
}
  
template<class scalar>
complex<scalar>
cos(const complex<scalar> &c)
{
   const scalar x = c.real();
   const scalar y = c.imag();
   return complex<scalar>(cos(x) * cosh(y), -sin(x) * sinh(y));
}

template<class scalar>
complex<scalar>
cosh(const complex<scalar> &c)
{
   const scalar x = c.real();
   const scalar y = c.imag();
   return complex<scalar>(cosh(x) * cos(y), sinh(x) * sin(y));
}

template<class scalar>
complex<scalar>
exp(const complex<scalar> &c)
{
   return polar<scalar>(exp(c.real()), c.imag());
}


template<class scalar>
complex<scalar>
log(const complex<scalar> &c)
{
   return complex<scalar>(log(abs(c)), arg(c));
}

template<class scalar>
complex<scalar>
log10(const complex<scalar> &c)
{
   return log(c)/log(scalar(10.0));
}

template<class scalar>
complex<scalar>
sin(const complex<scalar> &c)
{
   const scalar x = c.real();
   const scalar y = c.imag();
   return complex<scalar>(sin(x) * cosh(y), cos(x) * sinh(y)); 
}

template<class scalar>
complex<scalar>
sin(const complex<scalar> &c)
{
   const scalar x = c.real();
   const scalar y = c.imag();
   return complex<scalar>(sinh(x) * cos(y), cosh(x) * sin(y));
}

template<class scalar>
complex<scalar>
sqrt(const complex<scalar> &c)
{
   scalar temp = sqrt(abs(c));
   return polar<scalar>(temp, arg(c)/2);
}
  
template<class scalar>
complex<scalar>
tan(const complex<scalar> &c)
{
   return sin(c)/cos(c);
}
template<class scalar>
complex<scalar>
atan(const complex<scalar> &c)
{
   return cos(c)/sin(c);
}

template<class scalar>
complex<scalar>
tanh(const complex<scalar> &c)
{
   return sinh(c)/cosh(c);
}
template<class scalar>
complex<scalar>
atanh(const complex<scalar> &c)
{
   return cosh(c)/sinh(c);
}

template<class scalar>
complex<scalar>
pow(const complex<scalar> &x, int y)
{
   scalar temp = abs(x);
   temp = pow(temp,y);
   return polar<scalar>(temp,y*arg(x));
}

template<class scalar>
complex<scalar>
pow(const complex<scalar> &x, const scalar &y)
{
   return x==scalar() ? scalar() : polar<scalar>(pow(norm(x),y/2), y*arg(x));
   // this formula contains: pow(double), atan2(double,double), cos(double), sin(double)
    
   // other formula:
   // exp(y*log(x));
   // This would contain sqrt(double), log(double), atan2(double,double), exp(double)
   // sin(double), cos(double)
}

template<class scalar>
complex<scalar>
pow(const complex<scalar> &x, const complex<scalar> &y)
{
   return x==scalar() ? scalar() : exp(y*log(x));
}

template<class scalar>
complex<scalar>
pow(const scalar &x, const complex<scalar> &y)
{
   if(x == scalar()) return scalar();
   if(x  > scalar()) return exp(y*log(x));
   return exp(y*log(complex<scalar>(x,scalar())));
}


/**************************************************************************
* iostream
**************************************************************************/

template<class scalar>
ostream&
operator <<(ostream& ios,complex<scalar>& a)
{
  ios << "(" << a.real() << "," << a.imag() << ")" ;
  return(ios);
}

template<class scalar>
ostrstream& operator <<(ostrstream& ios,complex<scalar>& a)
{
  ios << "(" << a.real() << "," << a.imag() << ")" ;
  return(ios);
}

template<class scalar>
istream& operator >>(istream& ios,complex<scalar>& a)
{
  ios >> a.real() >> a.imag() ;
  return(ios);
}

template<class scalar>
istrstream& operator >>(istrstream& ios,complex<scalar>& a)
{
  ios >> a.real() >> b.imag() ;
  return(ios);
}


#endif
