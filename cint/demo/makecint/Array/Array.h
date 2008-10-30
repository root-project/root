/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* Array.h
*
* Array class template
*
**************************************************************************/

#ifndef ARRAY_H
#define ARRAY_H

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>

#include "Fundament.h"

// #define G__GARIONIS

#if defined(_WINDOWS) || defined(_WIN32) || defined(__GNUC__)
#define G__NOSTATICMEMBER
extern int G__defaultsize;
#endif

#if defined(_WINDOWS) || defined(_WIN32) 
#define G__NOFRIENDS
#endif

template <typename T> class Array;
template <typename T> Array<T> operator+(Array<T>& a,Array<T>& b);
template <typename T> Array<T> operator-(Array<T>& a,Array<T>& b);
template <typename T> Array<T> operator*(Array<T>& a,Array<T>& b);
template <typename T> Array<T> operator/(Array<T>& a,Array<T>& b);
template <typename T> Array<T> exp(Array<T>& a);
template <typename T> Array<T> abs(Array<T>& a);

/**********************************************************
* definition of array class
**********************************************************/

template<class T> class Array  {
public:
  Array(T start,T stop,int ndat);
  Array(T x);
  Array(Array<T> const & X);
  Array(void);
  Array(Array<T>& X,int offset,int ndat);
  ~Array(); 
  
  Array<T>& operator =(Array<T>& a);
  Array<T> operator()(int from,int to);
  T& operator[](int index);
  int getsize(void) { return(n); }
  int resize(int size);
  static void setdefaultsize(int size) { G__defaultsize = size; }

  // Ralf Garionis's bug report
  T*& newdat(T* d);

  void disp(ostream& ostr=std::cout);


#ifndef __CINT__
  friend Array<T> operator+(Array<T>& a,Array<T>& b);
  friend Array<T> operator-(Array<T>& a,Array<T>& b);
  friend Array<T> operator*(Array<T>& a,Array<T>& b);
  friend Array<T> operator/(Array<T>& a,Array<T>& b);
  friend Array<T> exp(Array<T>& a);
  friend Array<T> abs(Array<T>& a);
#endif

#ifdef NOT_READY_YET
  friend Array<T> log(Array<T>& a);
  friend Array<T> log10(Array<T>& a);
  friend Array<T> sinc(Array<T>& a);
  friend Array<T> sin(Array<T>& a);
  friend Array<T> cos(Array<T>& a);
  friend Array<T> tan(Array<T>& a);
  friend Array<T> asin(Array<T>& a);
  friend Array<T> acos(Array<T>& a);
  friend Array<T> atan(Array<T>& a);
  friend Array<T> sinh(Array<T>& a);
  friend Array<T> cosh(Array<T>& a);
  friend Array<T> tanh(Array<T>& a);
  friend Array<T> sqrt(Array<T>& a);
  friend Array<T> rect(Array<T>& a);
  friend Array<T> square(Array<T>& a);
  friend Array<T> rand(Array<T>& a);
  friend Array<T> conv(Array<T>& a,Array<T>& b);
  friend Array<T> integ(Array<T>& x,Array<T>& y);
  friend Array<T> diff(Array<T>& x,Array<T>& y);
#endif

#ifndef G__NOFRIENDS
private:
#endif
  int n;               // number of data
  T *dat;         // pointer to data Array
#ifndef G__NOSTATICMEMBER
  static int G__defaultsize;
#endif
  int malloced;
  enum { ISOLD, ISNEW };
  Array(T *p,int ndat,int isnew /* =0 */);

 public:
#if 0
  operator T () const;
  operator T* () const;
#endif
} ;

/***********************************************
* Ralf Garionis's bug report
***********************************************/
template <class T> T*& Array<T>::newdat(T* d)
{
  dat = d; 
  return dat;
}

#if 0
/***********************************************
* conversion (scalar type)
***********************************************/
template<class T> Array<T>::operator T () const {
  printf("###\n");
  return dat[0];
}

/***********************************************
* conversion (pointer type)
***********************************************/
template<class T> Array<T>::operator T* () const {
  return dat;
}
#endif

#ifndef G__NOSTATICMEMBER
template<class T> int Array<T>::G__defaultsize = 100;
#endif

/***********************************************
* Destructor
***********************************************/
template<class T> Array<T>::~Array()
{
  if(malloced) delete[] dat;
}

/***********************************************
* Copy constructor
***********************************************/
template<class T> Array<T>::Array(Array<T> const & X)
{
  if(X.malloced) {
    dat = new T[X.n];
    memcpy(dat,X.dat,X.n*sizeof(T));
    n = X.n;
    malloced=1;
  }
  else {
    dat=X.dat;
    n = X.n;
    malloced=0;
  }
}

/***********************************************
* Implicit conversion constructor 
***********************************************/
template<class T> Array<T>::Array(T x)
{
  n=G__defaultsize;
  dat = new T[n];
  malloced=1;
  for(int i=0;i<n;i++) dat[i] = x;
}


/***********************************************
* Constructor
***********************************************/
template<class T> Array<T>::Array(T start,T stop,int ndat)
{
  if(ndat<=0) {
    cerr << "Error: Size of array 0>=" << ndat 
         << ". default " << G__defaultsize << "used.\n";
  }
  else {
    G__defaultsize=ndat;
  }
  n = ndat;
  dat = new T[n];
  malloced=1;
  T res ;
  res = (stop-start)/(n-1);
  for(int i=0;i<n;i++) dat[i] = i*res + start;
}


/***********************************************
* constructor 
***********************************************/
template<class T> Array<T>::Array(void)
{
  n=G__defaultsize;
  dat = new T[n];
  malloced=1;
}

/***********************************************
* constructor 
***********************************************/
template<class T> Array<T>::Array(T *p,int ndat,int isnew)
{
  if(ndat<=0) {
    cerr << "Error: Size of array 0>=" << ndat 
         << ". default " << G__defaultsize << "used.\n";
  }
  else {
    G__defaultsize=ndat;
  }
  if(0==isnew) {
    dat = p;
    n = G__defaultsize;
    malloced=0;
  }
  else {
    n=G__defaultsize;
    dat = new T[ndat];
    malloced=1;
    memcpy(dat,p,ndat*sizeof(T));
  }
}


/***********************************************
* constructor for rvalue subArray
***********************************************/
template<class T> Array<T>::Array(Array<T>& X,int offset,int ndat)
{
  int i;
  if(offset<0||offset>=X.n) {
    cerr << "Illegal offset. Set to 0\n";
    offset = 0;
  }
  if(ndat<=0) {
    n=X.n-offset;
  }
  else {
    n = ndat;
  }
  dat = new T[n];
  malloced=1;
  if(offset+n>X.n) {
    memcpy(dat,X.dat+offset,(X.n-offset)*sizeof(T));
    for(i=X.n-offset;i<n;i++) dat[i] = 0;
  }
  else {
    memcpy(dat,X.dat+offset,n*sizeof(T));
  }
}

/***********************************************
* resize
***********************************************/
template<class T> int Array<T>::resize(int size)
{
  if(size<=0) {
    cerr << "Resize failed. Size of array 0>=" << size ;
  }
  else if(size!=n) {
    if(malloced) delete[] dat;
    n=size;
    malloced=1;
    dat=new T[n];
  }
  return(n);
}

/**********************************************************
* operator = as member function
**********************************************************/
template<class T> Array<T>& Array<T>::operator =(Array<T>& a)
{
  if(malloced && a.malloced) {
    if(a.n<n) memcpy(dat,a.dat,a.n*sizeof(T));
    else      memcpy(dat,a.dat,n*sizeof(T));
  }
  else {
    Array<T> c=Array<T>(a.dat,a.n,ISNEW);
    if(c.n<n) memcpy(dat,c.dat,c.n*sizeof(T));
    else      memcpy(dat,c.dat,n*sizeof(T));
  }
  return(*this);
}

/**********************************************************
* operator () as member function
**********************************************************/
template<class T> Array<T> Array<T>::operator()(int from,int to)
{
  if(from<0 || n<=to) {
    fprintf(stderr,"Error: Array index out of range (%d,%d),%d\n"
	    ,from,to,n);
    return(*this);
  }
  else {
    Array<T> c=Array<T>(dat+from,to-from+1,ISOLD);
    return(c);
  }
}

/**********************************************************
* operator [] as member function
**********************************************************/
template<class T> T& Array<T>::operator[](int index)
{
  if(index<0||n<=index) {
    fprintf(stderr,"Error: Array index out of range %d/%d\n"
	    ,index,n);
    return(dat[0]);
  }
  return(dat[index]);
}

/**********************************************************
* disp()
**********************************************************/
template<class T> void Array<T>::disp(ostream& ostr)
{
  ostr << "size=" << n << "\n";
  for(int i=0;i<n;i++) ostr << dat[i] << ' ' ;
  ostr << '\n';
}

/************************************************************************
* friend function
************************************************************************/

/***********************************************
* operator +
***********************************************/
template<class T> Array<T> operator +(Array<T>& a,Array<T>& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  Array<T> c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] + b.dat[i];
  return(c);
}

/***********************************************
* operator -
***********************************************/
template<class T> Array<T> operator-(Array<T>& a,Array<T>& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  Array<T> c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] - b.dat[i];
  return(c);
}

/***********************************************
* operator *
***********************************************/
template<class T> Array<T> operator*(Array<T>& a,Array<T>& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  Array<T> c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] * b.dat[i];
  return(c);
}

/***********************************************
* operator /
***********************************************/
template<class T> Array<T> operator /(Array<T>& a,Array<T>& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  Array<T> c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] / b.dat[i];
  return(c);
}

/**********************************************************
* class Array function overloading
**********************************************************/

/***********************************************
* exp
***********************************************/
template<class T> Array<T> exp(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = (T)exp(a[i]);
  return(c);
}

/***********************************************
* abs
***********************************************/
template<class T> Array<T> abs(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = (T)fabs(a[i]);
  return(c);
}


#ifndef NOT_READY_YET
/***********************************************
* log
***********************************************/
template<class T> Array<T> log(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = log(a[i]);
  return(c);
}

/***********************************************
* log10
***********************************************/
template<class T> Array<T> log10(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = log10(a[i]);
  return(c);
}

/***********************************************
* sinc
***********************************************/
template<class T> Array<T> sinc(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) {
    if(0==a[i]) c[i]=1;
    else        c[i] = sin(a[i]/a[i]);
  }
  return(c);
}


/***********************************************
* sin
***********************************************/
template<class T> Array<T> sin(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = sin(a[i]);
  return(c);
}

/***********************************************
* cos
***********************************************/
template<class T> Array<T> cos(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = cos(a[i]);
  return(c);
}

/***********************************************
* tan
***********************************************/
template<class T> Array<T> tan(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = tan(a[i]);
  return(c);
}

/***********************************************
* asin
***********************************************/
template<class T> Array<T> asin(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = asin(a[i]);
  return(c);
}

/***********************************************
* acos
***********************************************/
template<class T> Array<T> acos(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = acos(a[i]);
  return(c);
}

/***********************************************
* atan
***********************************************/
template<class T> Array<T> atan(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = atan(a[i]);
  return(c);
}


/***********************************************
* sinh
***********************************************/
template<class T> Array<T> sinh(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = sinh(a[i]);
  return(c);
}

/***********************************************
* cosh
***********************************************/
template<class T> Array<T> cosh(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = cosh(a[i]);
  return(c);
}

/***********************************************
* tanh
***********************************************/
template<class T> Array<T> tanh(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = tanh(a[i]);
  return(c);
}

/***********************************************
* sqrt
***********************************************/
template<class T> Array<T> sqrt(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = sqrt(a[i]);
  return(c);
}


/***********************************************
* rect
***********************************************/
template<class T> Array<T> rect(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = rect(a[i]);
  return(c);
}


/***********************************************
* square
***********************************************/
template<class T> Array<T> square(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = square(a[i]);
  return(c);
}

/***********************************************
* rand
***********************************************/
#undef rand
template<class T> Array<T> rand(Array<T>& a)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  for(int i=0;i<a.n;i++) c[i] = rand(a[i]);
  return(c);
}


/***********************************************
 * conv cross convolution
 ***********************************************/
template<class T> Array<T> conv(Array<T>& a,Array<T>& b)
{
  a.setdefaultsize(a.n);
  Array<T> c;
  int i,j,k;
  int f,t;
  f = b.n/2;
  t = b.n-f;
  for(i=0;i<a.n;i++) {
    c[i]=0;
    for(j=0;j<b.n;j++) {
      k=i-f+j;
      if(k<0)         c[i] += a[0]*b[j];
      else if(k>=a.n) c[i] += a[a.n-1]*b[j];
      else            c[i] += a[k]*b[j];
    }
  }
  return(c);
}

/***********************************************
 * integ
 ***********************************************/
template<class T> Array<T> integ(Array<T>& x,Array<T>& y)
{
  x.setdefaultsize(x.n);
  Array<T> c;
  int i;
  T integ=0;
  for(i=0;i<y.n-1;i++) {
    integ += y[i]*(x[i+1]-x[i]);
    c[i] = integ;
  }
  integ += y[i]*(x[i]-x[i-1]);
  c[i] = integ;
  return(c);
}

/***********************************************
 * diff differential
 ***********************************************/
template<class T> Array<T> diff(Array<T>& x,Array<T>& y)
{
  x.setdefaultsize(x.n);
  Array<T> c;
  int i;
  T integ=0;
  for(i=0;i<y.n;i++) {
    c[i] = (y[i+1]-y[i])/(x[i+1]-x[i]);
  }
  c[i] = c[i-1];
  return(c);
}
#endif

/**************************************************************************
* ostream
**************************************************************************/
template<class T> ostream& operator<<(ostream& ostr,Array<T>& x)
{
  x.disp(ostr);
  return(ostr);
}


/**************************************************************************
* int dummy
**************************************************************************/
Array<int> exp(Array<int>& a);


/**************************************************************************
* template instanciation
**************************************************************************/
typedef Array<int>     iarray;
typedef Array<double>  darray;
typedef Array<Complex> carray;


// ostream& operator<<(ostream& ostr,Array<int>& x);
// ostream& operator<<(ostream& ostr,Array<double>& x);
// ostream& operator<<(ostream& ostr,Array<Complex>& x);

#ifdef G__GARIONIS
// Ralf Garionis's bug report
template <class T> 
class Matrix : public Array<T> {
 public:
   Matrix() { }
   Matrix(double a) { }
   Matrix(int a) { }
   friend Matrix<T> operator-(Matrix<T>& a,Matrix<T>& b);
   friend Matrix<T> operator+(Matrix<T>& a,Matrix<T>& b);
   friend Matrix<T> operator*(Matrix<T>& a,Matrix<T>& b);
   friend Matrix<T> operator/(Matrix<T>& a,Matrix<T>& b);
};

template<class T> Matrix<T> operator-(Matrix<T>& a,Matrix<T>& b) 
{Matrix<T> c; return(c);}
template<class T> Matrix<T> operator+(Matrix<T>& a,Matrix<T>& b) 
{Matrix<T> c; return(c);}
template<class T> Matrix<T> operator*(Matrix<T>& a,Matrix<T>& b) 
{Matrix<T> c; return(c);}
template<class T> Matrix<T> operator/(Matrix<T>& a,Matrix<T>& b) 
{Matrix<T> c; return(c);}

typedef Matrix<int> imatrix;
typedef Matrix<double> dmatrix;
typedef Matrix<Complex> cmatrix;

// another bug report from Ralf Garionis
typedef Array<iarray> iiarray;
typedef Array<Array<int> > aiarray;
typedef Array<Matrix<int> > miarray;

#endif

#endif
