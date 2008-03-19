/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
 * fft.h
 *
 *  fft << time << x_in >> freq >> complex_out >> '\n';
 *  ifft << freq << complex_in >> time >> x_out >> '\n';
 *  spectrum << time << x_in >> freq >> spect_out >> '\n';
 *  cepstrum << time << x_in >> else freq >> cepst_out >> '\n';
*
**************************************************************************/
#ifndef G__FFT_H
#define G__FFT_H

#include <array.h>

#pragma security level0

#ifndef G__FFTSL
# ifdef G__SHAREDLIB
#pragma include_noerr <fft.dll>
#  ifndef G__FFTSL
#include <fft.c>
#  endif
# else
#include <fft.c>
# endif 
#endif // G__FFTSL


#define IDEN_FFT      1
#define IDEN_IFFT     2
#define IDEN_SPECTRUM 3
#define IDEN_CEPSTRUM 4

union G__FFT_DATA {
  carray *pc;
  array  *pa[2];
};

class G__FFT {
  int icount,ocount;
  int iden;

  array *xin;
  G__FFT_DATA yin;
  int iden_yin;

  array *xout;
  G__FFT_DATA yout;
  int iden_yout;

public:
  G__FFT(int identity); 
  G__FFT& operator <<(array& x);
  G__FFT& operator <<(carray& x);
  G__FFT& operator >>(array& x);
  G__FFT& operator >>(carray& x);
  G__FFT& operator <<(char c);
  G__FFT& operator <<(char *s);
  G__FFT& operator >>(char c);
  G__FFT& operator >>(char *s);
private:
  void sub_init(void);
  void do_fft(void);
  void sub_fft(void);
  void sub_ifft(void);
  void sub_spectrum(void);
  void sub_cepstrum(void);
};

G__FFT::G__FFT(int identity)
{ 
  iden=identity; 
  sub_init();
}

void G__FFT::sub_init(void)
{
  int i;
  xin=xout=(array*)NULL;
  for(i=0;i<2;i++) yin.pa[i]=yout.pa[i]=(array*)NULL;
  icount=ocount=0;
}


G__FFT& G__FFT::operator <<(array& x)
{
  switch(icount) {
  case 0:
    xin = &x;
    break;
  case 1:
    yin.pa[0] = &x;
    iden_yin=graphbuf::G__ARRAYTYPE;
    break;
  case 2:
    yin.pa[1] = &x;
    iden_yin=graphbuf::G__ARRAYTYPE;
    break;
  default:
    cerr << "Too many input is given\n";
    break;
  }
  ++icount;
  return(*this);
}

G__FFT& G__FFT::operator <<(carray& x)
{
  yin.pc = &x;
  iden_yin=graphbuf::G__CARRAYTYPE;
  return(*this);
}

G__FFT& G__FFT::operator >>(array& x)
{
  switch(ocount) {
  case 0:
    xout = &x;
    break;
  case 1:
    yout.pa[0] = &x;
    iden_yout=graphbuf::G__ARRAYTYPE;
    break;
  case 2:
    yout.pa[1] = &x;
    iden_yout=graphbuf::G__ARRAYTYPE;
    break;
  default:
    cerr << "Too many output is specified\n";
    break;
  }
  ++ocount;
  return(*this);
}

G__FFT& G__FFT::operator >>(carray& x)
{
  yout.pc = &x;
  iden_yout=graphbuf::G__CARRAYTYPE;
  return(*this);
}

G__FFT& G__FFT::operator >>(char *s)
{
  *this << '\n';
  return(*this);
}

G__FFT& G__FFT::operator >>(char c)
{
  *this << '\n';
  return(*this);
}
G__FFT& G__FFT::operator >>(G__CINT_ENDL c)
{
  *this << '\n';
  return(*this);
}
G__FFT& G__FFT::operator <<(G__CINT_ENDL c)
{
  *this << '\n';
  return(*this);
}

G__FFT& G__FFT::operator <<(char *s)
{
  *this << '\n';
  return(*this);
}

G__FFT& G__FFT::operator <<(char c)
{
  if(NULL!=yin.pa[0] && NULL!=yout.pa[0] &&
     NULL!=xin && NULL!=xout) {
    do_fft();
  }
  else {
    cerr << "Error: FFT not enough data or buffer given\n";
  }
  sub_init();
  return(*this);
}

void G__FFT::do_fft(void)
{
  switch(iden) {
  case IDEN_FFT:
    sub_fft();
    break;
  case IDEN_IFFT:
    sub_ifft();
    break;
  case IDEN_SPECTRUM:
    sub_spectrum();
    break;
  case IDEN_CEPSTRUM:
    sub_cepstrum();
    break;
  default:
    cerr << "Unknown FFT calculation\n";
    break;
  }
}

void G__FFT::sub_fft(void)
{
  if(graphbuf::G__CARRAYTYPE==iden_yout) {
    if(graphbuf::G__CARRAYTYPE==iden_yin) {
      *yout.pc = *yin.pc;
    }
    else {
      if(NULL==yin.pa[1]) {
	*yout.pc = *yin.pa[0];
      }
      else {
	*yout.pc=carray(*yin.pa[0],*yin.pa[1]);
      }
    }
    *xout = *xin;
    fft(*xout,*yout.pc);
  }
  else {
    if(NULL==yout.pa[1]) {
      cerr << "Error: FFT imaginary output not specified\n";
    }
    else {
      *xout = *xin;
      *yout.pa[0]=*yin.pa[0];
      if(yin.pa[1]) *yout.pa[1]=*yin.pa[1];
      else          *yout.pa[1]=0.0;
      fft(*xout,*yout.pa[0],*yout.pa[1]);
    }
  }
}

void G__FFT::sub_ifft(void)
{
  if(graphbuf::G__CARRAYTYPE==iden_yout) {
    if(graphbuf::G__CARRAYTYPE==iden_yin) {
      *yout.pc = *yin.pc;
    }
    else {
      if(NULL==yin.pa[1]) {
	*yout.pc = *yin.pa[0];
      }
      else {
	*yout.pc=carray(*yin.pa[0],*yin.pa[1]);
      }
    }
    *xout = *xin;
    ifft(*xout,*yout.pc);
  }
  else {
    if(NULL==yout.pa[1]) {
      *xout = *xin;
      *yout.pa[0]=*yin.pa[0];
      yout.pa[1]=new array(0,0,xout->n);
      if(yin.pa[1]) *yout.pa[1]=*yin.pa[1];
      else          *yout.pa[1]=0.0;
      ifft(*xout,*yout.pa[0],*yout.pa[1]);
      delete yout.pa[1];
    }
    else {
      *xout = *xin;
      *yout.pa[0]=*yin.pa[0];
      if(yin.pa[1]) *yout.pa[1]=*yin.pa[1];
      else          *yout.pa[1]=0.0;
      Ifft(*xout,*yout.pa[0],*yout.pa[1]);
    }
  }
}

void G__FFT::sub_spectrum(void)
{
  if(graphbuf::G__CARRAYTYPE==iden_yin || graphbuf::G__CARRAYTYPE==iden_yout) {
    cerr << "Error: spectrum, carray not expected\n";
  }
  else {
    *xout = *xin;
    *yout.pa[0] = *yin.pa[0];
    spectrum(*xout,*yout.pa[0]);
  }
}

void G__FFT::sub_cepstrum(void)
{
  if(graphbuf::G__CARRAYTYPE==iden_yin || graphbuf::G__CARRAYTYPE==iden_yout) {
    cerr << "Error: cepstrum, carray not expected\n";
  }
  else {
    *xout = *xin;
    *yout.pa[0]=*yin.pa[0];
    cepstrum(*xout,*yout.pa[0]);
  }
}


// Global FFT object
G__FFT fft=G__FFT(IDEN_FFT);
G__FFT ifft=G__FFT(IDEN_IFFT);
G__FFT spectrum=G__FFT(IDEN_SPECTRUM);
G__FFT cepstrum=G__FFT(IDEN_CEPSTRUM);

/**************************************************************************
*
**************************************************************************/

/* fft */
int fft(array& x,array& re,array& im)
{
#ifndef OLD
  if( x.n!=re.n) re.resize(x.n);
  if( x.n!=im.n ) im.resize(x.n);
#else
  if( x.n!=re.n || x.n!=im.n ) {
    cerr << "Error: size of array does not match\n";
    return(1);
  }
#endif
  fft(x.dat,re.dat,im.dat,x.n);
}

/* ifft */
int ifft(array& x,array& re,array& im)
{
#ifndef OLD
  if( x.n!=re.n) re.resize(x.n);
  if( x.n!=im.n ) im.resize(x.n);
#else
  if( x.n!=re.n || x.n!=im.n ) {
    cerr << "Error: size of array does not match\n";
    return(1);
  }
#endif
  ifft(x.dat,re.dat,im.dat,x.n);
}

/* spectrum */
int spectrum(array& x,array& re)
{
#ifndef OLD
  if( x.n!=re.n) re.resize(x.n);
#else
  if( x.n!=re.n ) {
    cerr << "Error: size of array does not match\n";
    return(1);
  }
#endif
  spectrum(x.dat,re.dat,x.n);
}

/* colleration (auto) */
int colleration(array& x,array& re)
{
#ifndef OLD
  if( x.n!=re.n) re.resize(x.n);
#else
  if( x.n!=re.n ) {
    cerr << "Error: size of array does not match\n";
    return(1);
  }
#endif
  colleration(x.dat,re.dat,x.n);
}

/* cepstrum */
int cepstrum(array& x,array& re)
{
#ifndef OLD
  if( x.n!=re.n) re.resize(x.n);
#else
  if( x.n!=re.n ) {
    cerr << "Error: size of array does not match\n";
    return(1);
  }
#endif
  cepstrum(x.dat,re.dat,x.n);
}


/* fft */
int fft(array& x,carray& y)
{
#ifndef OLD
  if( x.n!=y.n) y.resize(x.n);
#else
  if( x.n!=y.n ) {
    cerr << "Error: size of array does not match\n";
    return(1);
  }
#endif
  fft(x.dat,y.re,y.im,x.n);
}

/* ifft */
int ifft(array& x,carray& y)
{
#ifndef OLD
  if( x.n!=y.n) y.resize(x.n);
#else
  if( x.n!=y.n ) {
    cerr << "Error: size of array does not match\n";
    return(1);
  }
#endif
  ifft(x.dat,y.re,y.im,x.n);
}

#endif
