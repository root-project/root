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
   scalar real() { return re; }
   scalar imag() { return im; }
 protected:
  scalar re,im;
 public:
  friend complex operator +(complex& a,complex& b);
  friend complex operator -(complex& a,complex& b);
  friend complex operator *(complex& a,complex& b);
  friend complex operator /(complex& a,complex& b);
#ifndef __MAKECINT__
  friend complex exp(const complex& a);
#endif
  friend scalar abs(const complex& a);
};

template<class scalar>
complex<scalar> operator +(complex<scalar>& a,complex<scalar>& b)
{
  complex<scalar> c;
  c.re = a.re+b.re;
  c.im = a.im+b.im;
  return(c);
}
 
template<class scalar>
complex<scalar> operator -(complex<scalar>& a,complex<scalar>& b)
{
  complex<scalar> c;
  c.re = a.re-b.re;
  c.im = a.im-b.im;
  return(c);
}

template<class scalar>
complex<scalar> operator *(complex<scalar>& a,complex<scalar>& b)
{
  complex<scalar> c;
  c.re = a.re*b.re-a.im*b.im;
  c.im = a.re*b.im+a.im*b.re;
  return(c);
}

template<class scalar>
complex<scalar> operator /(complex<scalar>& a,complex<scalar>& b)
{
  complex<scalar> c;
  scalar x;
  x = b.re*b.re+b.im*b.im;
  c.re = (a.re*b.re+a.im*b.im)/x;
  c.im = (a.im*b.re-a.re*b.im)/x;
  return(c);
}


//**********************************************************************

#ifndef __MAKECINT__
template<class scalar>
complex<scalar> exp(const complex<scalar>& a)
{
  complex<scalar> c;
  scalar mag;
  mag = exp(a.real());
  c.re=mag*cos(a.im);
  c.im=mag*sin(a.im);
  return(c);
}
#endif

template<class scalar>
scalar abs(const complex<scalar>& a)
{
  scalar result;
  result = sqrt(a.re*a.re+a.im*a.im);
  return(result);
}



/**************************************************************************
* iostream
**************************************************************************/

template<class scalar>
ostream& operator <<(ostream& ios,complex<scalar>& a)
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
