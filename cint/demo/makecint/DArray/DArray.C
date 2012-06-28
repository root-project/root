/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* DArray.C
*
* Array class template
*
**************************************************************************/

#include "DArray.h"

using namespace std;

#ifdef __GNUC__
int G__defaultsize=100;
#else
int DArray::G__defaultsize = 100;
#endif


/***********************************************
* Destructor
***********************************************/
DArray::~DArray()
{
  if(malloced) delete[] dat;
}

/***********************************************
* Copy constructor
***********************************************/
DArray::DArray(DArray const & X)
{
  if(X.malloced) {
    dat = new double[X.n];
    memcpy(dat,X.dat,X.n*sizeof(double));
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
DArray::DArray(double x)
{
  n=G__defaultsize;
  dat = new double[n];
  malloced=1;
  for(int i=0;i<n;i++) dat[i] = x;
}


/***********************************************
* Constructor
***********************************************/
 DArray::DArray(double start,double stop,int ndat)
{
  if(ndat<=0) {
    cerr << "Error: Size of DArray 0>=" << ndat 
         << ". default " << G__defaultsize << "used.\n";
  }
  else {
    G__defaultsize=ndat;
  }
  n = ndat;
  dat = new double[n];
  malloced=1;
  double res ;
  res = (stop-start)/(n-1);
  for(int i=0;i<n;i++) dat[i] = i*res + start;
}


/***********************************************
* constructor 
***********************************************/
 DArray::DArray(void)
{
  n=G__defaultsize;
  dat = new double[n];
  malloced=1;
}

/***********************************************
* constructor 
***********************************************/
 DArray::DArray(double *p,int ndat,int isnew)
{
  if(ndat<=0) {
    cerr << "Error: Size of DArray 0>=" << ndat 
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
    dat = new double[ndat];
    malloced=1;
    memcpy(dat,p,ndat*sizeof(double));
  }
}


/***********************************************
* constructor for rvalue subDArray
***********************************************/
 DArray::DArray(DArray& X,int offset,int ndat)
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
  dat = new double[n];
  malloced=1;
  if(offset+n>X.n) {
    memcpy(dat,X.dat+offset,(X.n-offset)*sizeof(double));
    for(i=X.n-offset;i<n;i++) dat[i] = 0;
  }
  else {
    memcpy(dat,X.dat+offset,n*sizeof(double));
  }
}

/***********************************************
* resize
***********************************************/
 int DArray::resize(int size)
{
  if(size<=0) {
    cerr << "Resize failed. Size of DArray 0>=" << size ;
  }
  else if(size!=n) {
    if(malloced) delete[] dat;
    n=size;
    malloced=1;
    dat=new double[n];
  }
  return(n);
}

/**********************************************************
* operator = as member function
**********************************************************/
 DArray& DArray::operator =(DArray& a)
{
  if(malloced && a.malloced) {
    if(a.n<n) memcpy(dat,a.dat,a.n*sizeof(double));
    else      memcpy(dat,a.dat,n*sizeof(double));
  }
  else {
    DArray c=DArray(a.dat,a.n,ISNEW);
    if(c.n<n) memcpy(dat,c.dat,c.n*sizeof(double));
    else      memcpy(dat,c.dat,n*sizeof(double));
  }
  return(*this);
}

/**********************************************************
* operator () as member function
**********************************************************/
 DArray DArray::operator()(int from,int to)
{
  if(from<0 || n<=to) {
    fprintf(stderr,"Error: DArray index out of range (%d,%d),%d\n"
	    ,from,to,n);
    return(*this);
  }
  else {
    DArray c=DArray(dat+from,to-from+1,ISOLD);
    return(c);
  }
}

/**********************************************************
* operator [] as member function
**********************************************************/
 double& DArray::operator[](int index)
{
  if(index<0||n<=index) {
    fprintf(stderr,"Error: DArray index out of range %d/%d\n"
	    ,index,n);
    return(dat[0]);
  }
  return(dat[index]);
}

/**********************************************************
* disp()
**********************************************************/
 void DArray::disp(void)
{
  cout << "size=" << n << "\n";
  for(int i=0;i<n;i++) cout << dat[i] << ' ' ;
  cout << '\n';
}

/************************************************************************
* friend function
************************************************************************/

/***********************************************
* operator +
***********************************************/
 DArray operator +(DArray& a,DArray& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  DArray c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] + b.dat[i];
  return(c);
}

/***********************************************
* operator -
***********************************************/
 DArray operator-(DArray& a,DArray& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  DArray c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] - b.dat[i];
  return(c);
}

/***********************************************
* operator *
***********************************************/
 DArray operator *(DArray& a,DArray& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  DArray c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] * b.dat[i];
  return(c);
}

/***********************************************
* operator /
***********************************************/
 DArray operator /(DArray& a,DArray& b)
{
  int minn;
  if(a.n<b.n) minn = a.n;
  else        minn = b.n;
  a.setdefaultsize(minn);
  DArray c;
  for(int i=0;i<minn;i++) c.dat[i] = a.dat[i] / b.dat[i];
  return(c);
}

/**********************************************************
* class DArray function overloading
**********************************************************/

/***********************************************
* exp
***********************************************/
 DArray exp(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = (double)exp(a[i]);
  return(c);
}

/***********************************************
* abs
***********************************************/
 DArray abs(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = (double)fabs(a[i]);
  return(c);
}


#ifdef NOT_READY_YET
/***********************************************
* log
***********************************************/
 DArray log(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = log(a[i]);
  return(c);
}

/***********************************************
* log10
***********************************************/
 DArray log10(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = log10(a[i]);
  return(c);
}

/***********************************************
* sinc
***********************************************/
 DArray sinc(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) {
    if(0==a[i]) c[i]=1;
    else        c[i] = sin(a[i]/a[i]);
  }
  return(c);
}


/***********************************************
* sin
***********************************************/
 DArray sin(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = sin(a[i]);
  return(c);
}

/***********************************************
* cos
***********************************************/
 DArray cos(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = cos(a[i]);
  return(c);
}

/***********************************************
* tan
***********************************************/
 DArray tan(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = tan(a[i]);
  return(c);
}

/***********************************************
* asin
***********************************************/
 DArray asin(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = asin(a[i]);
  return(c);
}

/***********************************************
* acos
***********************************************/
 DArray acos(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = acos(a[i]);
  return(c);
}

/***********************************************
* atan
***********************************************/
 DArray atan(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = atan(a[i]);
  return(c);
}


/***********************************************
* sinh
***********************************************/
 DArray sinh(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = sinh(a[i]);
  return(c);
}

/***********************************************
* cosh
***********************************************/
 DArray cosh(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = cosh(a[i]);
  return(c);
}

/***********************************************
* tanh
***********************************************/
 DArray tanh(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = tanh(a[i]);
  return(c);
}

/***********************************************
* sqrt
***********************************************/
 DArray sqrt(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = sqrt(a[i]);
  return(c);
}


/***********************************************
* rect
***********************************************/
 DArray rect(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = rect(a[i]);
  return(c);
}


/***********************************************
* square
***********************************************/
 DArray square(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = square(a[i]);
  return(c);
}

/***********************************************
* rand
***********************************************/
 DArray rand(DArray& a)
{
  a.setdefaultsize(a.n);
  DArray c;
  for(int i=0;i<a.n;i++) c[i] = rand(a[i]);
  return(c);
}


/***********************************************
 * conv cross convolution
 ***********************************************/
 DArray conv(DArray& a,DArray& b)
{
  a.setdefaultsize(a.n);
  DArray c;
  int i,j,k;
  int f,t;
  f = b.n/2;
  t = b.n-f;
  for(i=0;i<n;i++) {
    c[i]=0;
    for(j=0;j<m;j++) {
      k=i-f+j;
      if(k<0)       c[i] += a[0]*b[j];
      else if(k>=n) c[i] += a[n-1]*b[j];
      else          c[i] += a[k]*b[j];
    }
  }
  return(c);
}

/***********************************************
 * integ
 ***********************************************/
 DArray integ(DArray& x,DArray& y)
{
  a.setdefaultsize(a.n);
  DArray c;
  int i;
  double integ=0;
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
 DArray diff(DArray& x,DArray& y)
{
  a.setdefaultsize(a.n);
  DArray c;
  int i;
  double integ=0;
  for(i=0;i<y.n;i++) {
    c[i] = (y[i+1]-y[i])/(x[i+1]-x[i]);
  }
  c[i] = c[i-1];
  return(c);
}
#endif

