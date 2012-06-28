/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
 * fft.c
 *
 *  makecint -A -dl fft.sl -c fft.c
 **************************************************************************/
#include <math.h>
#include <stdio.h>

#define FFT_MAX 4096

#define G__FFTSL
/* #define G__STDFFT */


/*******************************************************************/
/*  dft_cal.c         1996   1/8           M.Goto                  */
/*******************************************************************/

int dft_cal(double *x,double *re,double *im,int ndat)
{
  int i,j,m;
  double *s,*c;
  double *rei,*imi;
  double a,b;
  double sums,sumc;

  fprintf(stderr,"original data points=%d   DFT data points=%d\n",ndat,ndat); 

  s = (double*)malloc(sizeof(double)*(ndat));
  c = (double*)malloc(sizeof(double)*(ndat));
  rei = (double*)malloc(sizeof(double)*(ndat));
  imi = (double*)malloc(sizeof(double)*(ndat));

  /* prepare sin,cos table and copy input data */
  a=0 ;
  b=3.141592*2/ndat;
  for(i=0;i<ndat;i++) {
    s[i] = sin(a);
    c[i] = cos(a);
    /* printf("%d a=%g cos=%g sin=%g\n",i,a,c[i],s[i]); */
    a += b;
    rei[i] = re[i];
    imi[i] = im[i];
  }

  /* DFT calculation */
  for(i=0;i<ndat;i++) {
    sums=sumc=0.0;
    for(j=0;j<ndat;j++) {
      m = (j*i)%ndat;
      sumc += c[m]*rei[j] + s[m]*imi[j];
      sums += c[m]*imi[j] - s[m]*rei[j];
    }
    re[i] = sumc;
    im[i] = sums;
  }

  free((void*)imi);
  free((void*)rei);
  free((void*)c);
  free((void*)s);

  return(0);
}

/*******************************************************************/
/*  fft_cal.c         1987   1/8           M.Goto                  */
/*******************************************************************/

int fft_cal(double *x,double *re,double *im,int ndat)
{
  int g,h,i,j,k,l,m,n,p,q;
  /* double a,b,tmp,*pd,s[FFT_MAX],c[FFT_MAX]; */
  double a,b,tmp,*s,*c;
	
	
  /*  ndat  ----- must be     n=2**m        */
  n=2; 
  m=1;
  while (ndat  >   n) {   
    n *=  2 ;
    m++ ;
  }

  if(ndat!=n) {
    dft_cal(x,re,im,ndat);
    return(0);
  }

  fprintf(stderr
	  ,"original data points=%d   FFT data points=%d  m=%d n=2**m \n"
	  , ndat , n , m ); 
  /* ndat=n/2; */

  s = (double*)malloc(sizeof(double)*(n/2+1));
  c = (double*)malloc(sizeof(double)*(n/2+1));
  
  /* initialize sin cos table */
  a=0 ;
  b=3.141592*2/n;
  for(i=0;i<=n/2;i++) {
    s[i] = sin(a);
    c[i] = cos(a);
    /* c[i] = sqrt(1-s[i]*s[i]); */
    a += b;
  }
  
  /* Butterfly */
  l=n; 
  h=1;
  for (g=1;g<=m;g++) { 
    l /= 2 ;
    k=0 ;
    for (q=1;q<=h;q++) {
      p=0 ;
      for (i=k;i<=l+k-1;i++) {
	j=i+l ;
	a = re[i] - re[j];
	b = im[i] - im[j];
	re[i] = re[i] + re[j];
	im[i] = im[i] + im[j];
	if (p  ==  0) {  
	  re[j] = a;
	  im[j] = b;
	}
	else {
	  re[j] = a*c[p] + b*s[p];
	  im[j] = b*c[p] - a*s[p];
	}
	p=p+h;
      }
      k=k+l+l;
    }
    h=h+h;
  }

  free((void*)c);
  free((void*)s);
  
  /* Bit Reversal */
  for (i=1;i<=n-1;i++) {
    q=i;
    k=n/2;
    j=0;
    for (l=1;l<=m;l++) {
      j=j+k*(q%2);        /* j=j+k*mod(q,2) */
      q /= 2;              /* q=int(q/2)     */
      k /= 2;
    }
    if (i   <  j) {
      tmp= re[i];
      re[i]=re[j];
      re[j]=tmp;
      tmp=im[i];
      im[i]=im[j];
      im[j]=tmp;
    }
  }
  
  /* compen */
  tmp= re[n/2-1]; 
  re[n/2-1] = re[n/2+1] ;
  re[n/2+1] = tmp;
  tmp= im[n/2-1];
  im[n/2-1]=im[n/2+1];
  im[n/2+1]=tmp;

  return(0);
}   

/*******************************************************************/
/*  fft.c             1987   1/8           M.Goto                  */
/*******************************************************************/

int fft(double *x,double *re,double *im,int ndat)
{
  int i;
  double dT,T,dfreq;
  
  fft_cal(x,re,im,ndat);
  
  dT= x[1]-x[0];
  T = ndat * dT ;
  dfreq	= 1 / T ;
  
  for (i=0;i<ndat;i++) {
    x[i]  = dfreq * i ;
#ifndef G__STDFFT
    re[i] /= ndat;
    im[i] /= ndat;
#endif
  }
  return(0);
}

/*******************************************************************/
/*  ifft.c            1987   1/8           M.Goto                  */
/*******************************************************************/

int ifft(double *x,double *re,double *im,int ndat)
{
  int i,j;
  double dT,freq,dfreq,tmp;
  
  fft_cal(x,re,im,ndat);
  
  dfreq= x[1]-x[0];
  freq = ndat * dfreq ;
  dT = 1 / freq ;
  
  for(i=0;i<ndat;i++) {
    x[i] = dT * i;
  }
  
  /****************  Time reversal *********************/
  
  for (i=1;i<ndat/2-1;i++) {
    j= ndat - i ;
    tmp = re[i];
    re[i]=re[j];
#ifndef G__STDFFT
    re[j]=tmp;
#else
    re[j]=tmp/ndat;
#endif
    tmp = im[i];
    im[i]=im[j];
#ifndef G__STDFFT
    im[j]=tmp;
#else
    im[j]=tmp/ndat;
#endif
  }
  return(0);
}

/*******************************************************************/
/*  spectrum.c        1987   1/8           M.Goto                  */
/*******************************************************************/

int spectrum(double *x,double *y,int ndat)
{
  int i;
  
  double *im;
  
  im = (double*)calloc(ndat,sizeof(double));
  
  fft(x,y,im,ndat);
  
#ifndef G__STDFFT
  for (i=0;i<ndat;i++) y[i] = sqrt(y[i]*y[i] + im[i]*im[i]) * 1.41421356;
#else
  double k = 1.41421356/ndat;
  for (i=0;i<ndat;i++) y[i] = sqrt(y[i]*y[i] + im[i]*im[i]) * k;
#endif

  free((void*)im);
  return(0);
}

/*******************************************************************/
/*  colleration.c     1987   1/8           M.Goto                  */
/*******************************************************************/

int colleration(double *x,double *y,int ndat)
{
  /* auto colleration */
  int i;
  double *im;
  double tmp;
  
  spectrum(x,y,ndat);
  
  im = (double*)calloc(ndat,sizeof(double));
  
  ifft(x,y,im,ndat);
  
  tmp=y[0];
  
  for (i=0;i<ndat;i++) y[i] = y[i]/tmp; 

  free(im);

  return(0);
}

/*******************************************************************/
/*  cepstrum.c        1987   1/8           M.Goto                  */
/*******************************************************************/

int cepstrum(double *x,double *y,int ndat)
{
  int i;
  
  spectrum(x,y,ndat);
  
  for (i=0;i<ndat;i++) y[i] = log(y[i]);
  
  spectrum(x,y,ndat);
  return(0);
}

