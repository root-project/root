/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* darray.h
*
* Array class
*
*  Constructor, copy constructor, destructor, operator overloading
* function overloading, reference type
*
**************************************************************************/

#ifndef G__DARRAY_H
#define G__DARRAY_H

/**********************************************************
* definition of array class
**********************************************************/

int G__arraysize = 100;

class array  {
  int malloced;
 public:
  double *dat;         // pointer to data array
  int n;               // number of data

  array(double start,double stop,int ndat);
  array(double x);
  array(array& X);
  array(void);
  array(array& X,int offset,int ndat);
  array(double *p,int ndat,int isnew=0);
  ~array(); 

  array& operator =(array& a);

  array operator()(int from,int to);
  double& operator[](int index);

  int resize(int size);
  int getsize() { return(n); }
} ;


/***********************************************
* Destructor
***********************************************/
array::~array()
{
  if(malloced) {
    delete[] dat;
  }
}

/***********************************************
* Copy constructor
***********************************************/
array::array(array& X)
{
  int i;
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
array::array(double x)
{
  if(G__arraysize==0) {
    cerr << "Error: Size of array 0\n";
    return;
  }
  dat = new double[G__arraysize];
  G__ary_assign(dat,x,x,G__arraysize);
  n=G__arraysize;
  malloced=1;
}


/***********************************************
* Constructor
***********************************************/
array::array(double start,double stop,int ndat)
{
  double res;
  G__arraysize=ndat;
  dat = new double[G__arraysize];
  G__ary_assign(dat,start,stop,G__arraysize);
  n = G__arraysize;
  malloced=1;
}

/***********************************************
* constructor 
***********************************************/
array::array(void)
{
  if(G__arraysize==0) {
    cerr << "Error: Size of array 0\n";
    return;
  }
  dat = new double[G__arraysize];
  n=G__arraysize;
  malloced=1;
}

/***********************************************
* constructor 
***********************************************/
array::array(double *p,int ndat,int isnew)
{
  G__arraysize=ndat;
  if(isnew==0) {
    dat = p;
    n = G__arraysize;
    malloced=0;
  }
  else {
    dat = new double[ndat];
    memcpy(dat,p,ndat*sizeof(double));
    n=G__arraysize;
    malloced=1;
  }
}


/***********************************************
* constructor for rvalue subarray
***********************************************/
array::array(array& X,int offset,int ndat)
{
  int i;
  dat = new double[ndat];
  if(offset+ndat>X.n) {
    memcpy(dat,X.dat+offset,(X.n-offset)*sizeof(double));
    for(i=X.n-offset;i<ndat;i++) dat[i] = 0.0;
  }
  else {
    memcpy(dat,X.dat+offset,ndat*sizeof(double));
  }
  n = ndat;
  malloced=1;
}

/***********************************************
* resize
***********************************************/
int array::resize(int size)
{
  double *temp;
  if(size<n) {
    temp = new double[size];
    memcpy(temp,dat,sizeof(double)*size);
    if(malloced) delete[] dat;
    dat=temp;
    n=size;
    malloced=1;
  }
  else if(size>n) {
    temp = new double[size];
    memset(temp,0,sizeof(double)*size);
    memcpy(temp,dat,sizeof(double)*n);
    if(malloced) delete[] dat;
    dat=temp;
    n=size;
    malloced=1;
  }
  return(n);
}

/**********************************************************
* operator = as member function
**********************************************************/
array& array::operator =(array& a)
{
  int i;
  if(malloced && a.malloced) {
    if(a.n<n) memcpy(dat,a.dat,a.n*sizeof(double));
    else      memcpy(dat,a.dat,n*sizeof(double));
  }
  else {
    array c=array(a.dat,a.n,1);
    if(c.n<n) memcpy(dat,c.dat,c.n*sizeof(double));
    else      memcpy(dat,c.dat,n*sizeof(double));
  }
  return(*this);
}

/**********************************************************
* operator () as member function
**********************************************************/
array array::operator()(int from,int to)
{
  if(from<0 || n<=to) {
    fprintf(stderr,"Error: array index out of range %(d,%d),%d\n"
	    ,from,to,n);
    return(*this);
  }
  else {
    array c=array(dat+from,to-from+1,0);
    return(c);
  }
}

/**********************************************************
* operator [] as member function
**********************************************************/
double& array::operator[](int index)
{
  if(index<0||n<=index) {
    fprintf(stderr,"Error: array index out of range %d/%d\n"
	    ,index,n);
    return(dat[0]);
  }
  return(dat[index]);
}


/***********************************************
* operator +
***********************************************/
array operator +(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_plus(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator -
***********************************************/
array operator -(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_minus(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator -
***********************************************/
array operator -(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  array b=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_minus(c.dat,b.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator *
***********************************************/
array operator *(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_multiply(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator /
***********************************************/
array operator /(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_divide(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator @ (power)
***********************************************/
array operator @(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_power(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator << (shift)
***********************************************/
array operator <<(array& a,int shift)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  for(i=0;i<a.n-shift;i++) {c.dat[i] = a.dat[i+shift] ;}
  c.n=a.n;
  return(c);
}

/***********************************************
* operator >> (shift)
***********************************************/
array operator >>(array& a,int shift)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  for(i=0;i<a.n-shift;i++) {c.dat[i+shift] = a.dat[i] ;}
  c.n=a.n;
  return(c);
}


/**********************************************************
* class array function overloading
**********************************************************/

/***********************************************
* exp
***********************************************/
array exp(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_exp(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* log
***********************************************/
array log(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_log(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* log10
***********************************************/
array log10(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_log10(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* sinc
***********************************************/
array sinc(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sinc(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* sin
***********************************************/
array sin(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sin(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* cos
***********************************************/
array cos(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_cos(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* tan
***********************************************/
array tan(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_tan(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* sinh
***********************************************/
array sinh(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sinh(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* cosh
***********************************************/
array cosh(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_cosh(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* tanh
***********************************************/
array tanh(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_tanh(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* asin
***********************************************/
array asin(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_asin(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* acos
***********************************************/
array acos(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_acos(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* atan
***********************************************/
array atan(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_atan(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * abs
 ***********************************************/
array abs(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_fabs(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * fabs
 ***********************************************/
array fabs(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_fabs(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * sqrt
 ***********************************************/
array sqrt(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sqrt(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * rect
 ***********************************************/
array rect(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_rect(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * square
 ***********************************************/
array square(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_square(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
  * rand
  ***********************************************/
array rand(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_rand(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}


/***********************************************
 * conv cross convolution
 ***********************************************/
array conv(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_conv(c.dat,a.dat,a.n,b.dat,b.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * integ
 ***********************************************/
array integ(array& a,array& b) // a : x , b : y
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_integ(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * diff differential
 ***********************************************/
array diff(array& a,array& b) // a : x , b : y
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_diff(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * max
 ***********************************************/
array max(array& a,array& b) // a : x , b : y
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_max(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * min
 ***********************************************/
array min(array& a,array& b) // a : x , b : y
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_min(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* pow
***********************************************/
array pow(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_power(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* FIT filter
***********************************************/
array fir(array& in,array& filter) 
{
  // convolution is done in conv(). This function
  // simply does energy normalization 
  array out(0.0,0.0,in.getsize());
  int i,j,k;
  array fil=array(0.0,0.0 ,filter.n);
  double sum=0;
  k=fil.n;
  for(i=0;i<k;i++) sum += filter.dat[i];
  fil = filter/sum;
  out=conv(in,fil);
  return(out);
}

//////////////////////////////////////////////////////////

#ifndef G__ARRAY_H
#include <array.h>
#endif

#ifdef __CINT__
int G__ateval(const array& x) {
  int n = x.getsize();
  cout << "(array " << &x << ")" ;
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

