/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* carray.h
*
* Array class
*
*  Constructor, copy constructor, destructor, operator overloading
* function overloading, reference type
*
**************************************************************************/


#ifndef G__CARRAY_H
#define G__CARRAY_H

#ifndef G__ARRAY_H
class array;
#endif

#ifndef G__COMPLEX_H
class complex;
#endif


/**********************************************************
* definition of array class
**********************************************************/

// int G__arraysize = 100; // already declared in darray.h

class carray  {
 public:
  double *re,*im;      // pointer to data array
  int n;               // number of data

  //allocation
  carray(double real,double imag,int ndat);
  carray(void);

  //conversion 
  carray(complex& x);
  carray(double x,double y=0);
  carray(array& X);
  carray(carray& X);

  ~carray(); 

  carray& operator =(carray& a);
  carray operator()(int from,int to);
  complex& operator[](int index);

  int resize(int size);
  int getsize() { return(n); }
} ;

/***********************************************
* Destructor
***********************************************/
carray::~carray()
{
  delete[] re;
  delete[] im;
}

/***********************************************
* Copy constructor
***********************************************/
carray::carray(carray& X)
{
  int i;
  re = new double[X.n];
  im = new double[X.n];
  memcpy(re,X.re,X.n*sizeof(double));
  memcpy(im,X.im,X.n*sizeof(double));
  n = X.n;
}

/***********************************************
* Implicit conversion constructor 
***********************************************/

// double to carray
carray::carray(double x,double y)
{
  if(G__arraysize==0) {
    cerr << "Error: Size of carray 0\n";
    return;
  }
  re = new double[G__arraysize];
  im = new double[G__arraysize];
  G__ary_assign(re,x,x,G__arraysize);
  G__ary_assign(im,y,y,G__arraysize);
  n=G__arraysize;
}

// complex to carray
carray::carray(complex& x)
{
  if(G__arraysize==0) {
    cerr << "Error: Size of carray 0\n";
    return;
  }
  re = new double[G__arraysize];
  im = new double[G__arraysize];
  G__ary_assign(re,x.re,x.re,G__arraysize);
  G__ary_assign(im,x.im,x.im,G__arraysize);
  n=G__arraysize;
}

// array to carray
carray::carray(array& X)
{
  int i;
  re = new double[X.n];
  im = new double[X.n];
  memcpy(re,X.dat,X.n*sizeof(double));
  G__ary_assign(im,0.0,0.0,X.n);
  n = X.n;
}

// 2 arrays to carray
carray::carray(array& X,array& Y)
{
  int i;
  re = new double[X.n];
  im = new double[X.n];
  memcpy(re,X.dat,X.n*sizeof(double));
  if(Y.n!=X.n) {
    cerr << "Size unmatch\n";
  }
  else {
    memcpy(im,Y.dat,X.n*sizeof(double));
  }
  n = X.n;
}



/***********************************************
* Constructor
***********************************************/
carray::carray(double real,double imag,int ndat)
{
  double res;
  G__arraysize=ndat;
  re = new double[G__arraysize];
  im = new double[G__arraysize];
  G__ary_assign(re,real,real,G__arraysize);
  G__ary_assign(im,imag,imag,G__arraysize);
  n = G__arraysize;
}

/***********************************************
* constructor 
***********************************************/
carray::carray(void)
{
  if(G__arraysize==0) {
    cerr << "Error: Size of array 0\n";
    return;
  }
  re = new double[G__arraysize];
  im = new double[G__arraysize];
  n=G__arraysize;
}

/***********************************************
* constructor 
***********************************************/
carray::carray(double *pre,double *pim,int ndat)
{
  G__arraysize=ndat;
  re = new double[G__arraysize];
  im = new double[G__arraysize];
  memcpy(re,pre,ndat*sizeof(double));
  memcpy(im,pim,ndat*sizeof(double));
  n = G__arraysize;
}

/***********************************************
* constructor for rvalue subarray
***********************************************/
carray::carray(carray& X,int offset,int ndat)
{
  int i;
  re = new double[ndat];
  im = new double[ndat];
  if(offset+ndat>X.n) {
    memcpy(re,X.re+offset,(X.n-offset)*sizeof(double));
    memcpy(im,X.im+offset,(X.n-offset)*sizeof(double));
    for(i=X.n-offset;i<ndat;i++) {
      re[i] = 0.0;
      im[i] = 0.0;
    }
  }
  else {
    memcpy(re,X.re+offset,ndat*sizeof(double));
    memcpy(im,X.im+offset,ndat*sizeof(double));
  }
  n = ndat;
}


/***********************************************
* resize
***********************************************/
int carray::resize(int size)
{
  double *tempre,*tempim;
  
  if(size<n) {
    tempre = new double[size];
    tempim = new double[size];
    memcpy(tempre,re,sizeof(double)*size);
    memcpy(tempim,im,sizeof(double)*size);
    delete[] re;
    delete[] im;
    re=new double[size];
    im=new double[size];
    memcpy(re,tempre,sizeof(double)*size);
    memcpy(im,tempim,sizeof(double)*size);
    delete[] tempre;
    delete[] tempim;
    n=size;
  }
  else if(size>n) {
    tempre = new double[n];
    tempim = new double[n];
    memcpy(tempre,re,sizeof(double)*n);
    memcpy(tempim,im,sizeof(double)*n);
    delete[] re;
    delete[] im;
    re=new double[size];
    im=new double[size];
    memcpy(re,tempre,sizeof(double)*n);
    memcpy(im,tempim,sizeof(double)*n);
    delete[] tempre;
    delete[] tempim;
    n=size;
  }
  return(n);
}


/**********************************************************
* operator = as member function
**********************************************************/
carray& carray::operator =(carray& a)
{
  int i;
  if(a.n<n) {
    memcpy(re,a.re,a.n*sizeof(double));
    memcpy(im,a.im,a.n*sizeof(double));
  }
  else {
    memcpy(re,a.re,n*sizeof(double));
    memcpy(im,a.im,n*sizeof(double));
  }
  return(*this);
}


/**********************************************************
* operator () as member function
**********************************************************/
carray carray::operator()(int from,int to)
{
  if(from<0 || n<=to) {
    fprintf(stderr,"Error: array index out of range %(d,%d),%d\n"
	    ,from,to,n);
    return(*this);
  }
  else {
    carray c=carray(re+from,im+from,to-from+1,0);
    return(c);
  }
}

/**********************************************************
* operator [] as member function
**********************************************************/
complex carray::operator[](int index)
{
  complex c;
  if(index<0||n<=index) {
    fprintf(stderr,"Error: array index out of range %d/%d\n"
	    ,index,n);
    index = 0;
  }
  c.re = re[index];
  c.im = im[index];
  return(c);
}



/***********************************************
* operator +
***********************************************/
carray operator +(carray& a,carray& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  int i;
  G__ary_plus(c.re,a.re,b.re,a.n);
  G__ary_plus(c.im,a.im,b.im,a.n);
  c.n=a.n;
  return(c);
}

#if (G__CINTVERSION<5014035)
carray operator +(array& a,complex& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A+B;
  return(c);
}
carray operator +(complex& a,array& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A+B;
  return(c);
}
#endif


/***********************************************
* operator -
***********************************************/
carray operator -(carray& a,carray& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  int i;
  G__ary_minus(c.re,a.re,b.re,a.n);
  G__ary_minus(c.im,a.im,b.im,a.n);
  c.n=a.n;
  return(c);
}

carray operator -(carray& a)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray b=carray(0.0 , 0.0 , a.n);
  int i;
  G__ary_minus(c.re,b.re,a.re,a.n);
  G__ary_minus(c.im,b.im,a.im,a.n);
  c.n=a.n;
  return(c);
}

#if (G__CINTVERSION<5014035)
carray operator -(array& a,complex& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A-B;
  return(c);
}
carray operator -(complex& a,array& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A-B;
  return(c);
}
#endif


/***********************************************
* operator *
***********************************************/
carray operator *(carray& a,carray& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  int i;
  G__cary_multiply(c.re,c.im,a.re,a.im,b.re,b.im,a.n);
  c.n=a.n;
  return(c);
}
#if (G__CINTVERSION<5014035)
carray operator *(array& a,complex& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A*B;
  return(c);
}
carray operator *(complex& a,array& b)
{
  carray c=carray(0.0 , 0.0 , b.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A*B;
  return(c);
}
#endif


/***********************************************
* operator /
***********************************************/
carray operator /(carray& a,carray& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  int i;
  G__cary_divide(c.re,c.im,a.re,a.im,b.re,b.im,a.n);
  c.n=a.n;
  return(c);
}

#if (G__CINTVERSION<5014035)
carray operator /(array& a,complex& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A/B;
  return(c);
}
carray operator /(complex& a,array& b)
{
  carray c=carray(0.0 , 0.0 , a.n);
  carray A=carray(a);
  carray B=carray(b);
  c=A/B;
  return(c);
}
#endif


/***********************************************
* operator << (shift)
***********************************************/
carray operator <<(carray& a,int shift)
{
  carray c=carray(0.0 , 0.0 , a.n);
  int i;
  for(i=0;i<a.n-shift;i++) {
    c.re[i] = a.re[i+shift] ;
    c.im[i] = a.im[i+shift] ;
  }
  c.n=a.n;
  return(c);
}

/***********************************************
* operator >> (shift)
***********************************************/
carray operator >>(carray& a,int shift)
{
  carray c=carray(0.0 , 0.0 , a.n);
  int i;
  for(i=0;i<a.n-shift;i++) {
    c.re[i+shift] = a.re[i] ;
    c.im[i+shift] = a.im[i] ;
  }
  c.n=a.n;
  return(c);
}


/**********************************************************
* class array function overloading
**********************************************************/

/***********************************************
* exp
***********************************************/
carray exp(carray& a)
{
  carray c=carray(0.0 , 0.0 , a.n);
  int i;
  G__cary_exp(c.re,c.im,a.re,a.im,a.n);
  c.n=a.n;
  return(c);
}


/***********************************************
 * abs
 ***********************************************/
array abs(carray& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__cary_fabs(c.dat,a.re,a.im,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * re
 ***********************************************/
array re(carray& X)
{
  int i,n;
  array c=array(0.0,0.0,X.n);
  n=X.n;
  memcpy(c.dat,X.re,X.n*sizeof(double));
  c.n = X.n;
  return(c);
}

/***********************************************
 * im 
 ***********************************************/
array im(carray& X)
{
  array c=array(0.0,0.0,X.n);
  memcpy(c.dat,X.im,X.n*sizeof(double));
  c.n = X.n;
  return(c);
}


/***********************************************
 * db
 ***********************************************/
array db(carray& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__cary_fabs(c.dat,a.re,a.im,a.n);
  c.n=a.n;
  c = 20*log10(c);
  return(c);
}

/***********************************************
 * phase
 ***********************************************/
array phase(carray& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_divide(c.dat,a.im,a.re,a.n);
  G__ary_atan(c.dat,c.dat,a.n);
  c.n=a.n;
  c = c*180/3.141592;
  for(i=0;i<c.n;i++) {
    if(a.re[i]<0) c.dat[i] = c.dat[i]-180;
    if(i && c.dat[i-1]>80 && c.dat[i]<-200) c.dat[i] = c.dat[i] + 360;
  }
  return(c);
}

/***********************************************
 * parallel
 ***********************************************/
carray parallel(const carray& a,const carray& b) {
  carray c = a*b/(a+b);
  return(c);
}

/***********************************************
 * phase margin
 ***********************************************/
#include <utility>
pair<double,double> phasemargin(array f,array amp,array p) {
  int n = amp.getsize();
  pair<double,double> result;
  for(int i=0;i<n;i++) {
    if(amp[i]<0) {
      if(i==0) {
        result.first  = p[i];
        result.second = f[i];
      }
      else {
        result.first  = p[i] - (p[i-1]-p[i])*amp[i]/(amp[i-1]-amp[i]);
        result.second = f[i] - (f[i-1]-f[i])*amp[i]/(amp[i-1]-amp[i]);
      }
      return(result);
    }
  }
  cerr << "!!! Error : can not find 0dB cross" << endl;
  result.first  = 99e99;
  result.second = 99e99;
  return(result);
}

/***********************************************
 * phase margin
 ***********************************************/
double phasemargin(array amp,array p) {
  int n = amp.getsize();
  double pm;
  for(int i=0;i<n;i++) {
    if(amp[i]<0) {
      if(i==0) {
        pm = p[i];
      }
      else {
        pm = p[i] - (p[i-1]-p[i])*amp[i]/(amp[i-1]-amp[i]);
      }
      return(pm);
    }
  }
  cerr << "!!! Error : can not find 0dB cross" << endl;
  return(99e99);
}


#ifndef G__ARRAY_H
#include <array.h>
#endif

#ifdef __CINT__
int G__ateval(const carray& x) {
  int n = x.getsize();
#ifdef G__DISPALL
  for(int i=0;i<n-1;i++) cout << x[i] << ",";
#else
  if(n>20) {
    for(int i=0;i<10;i++) cout << x[i] << ",";
    cout << ",,,";
    for(int i=n-10;i<n-1;i++) cout << x[i] << ",";
  }
  else for(int i=0;i<n-1;i++) cout << x[i] << ",";
#endif
  cout << x[n-1] << endl;
  return(1); 
}
#endif

#endif
