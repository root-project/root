/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/******************************************************
* fir.c
*
*  FIR filter simulation
*
* array and fft pre-compiled class library is needed to run 
* this demo program.
******************************************************/
#include <array.h>
#include <fft.h>

const double PI=3.141592;

main()
{
	double w1=2*PI*50e6;
	double w2=2*PI*3e6;

	array filter=array(-3.0*PI ,3.0*PI ,50); 
	filter = sinc(filter); // FIR transient response

	array time=array(0.0 ,1e-6 ,512); //start,stop,datapoint
	array in,out,freq;
	array fftin,fftout;

	// input signal calculation
	in = sin(w1*time)*0.4+sin(w2*time);

	// FIR filter
	fir(in,filter,out);

	// plot time domain signal
	plot << "FIR filter"
             << time  << "sec"
	     << in    << "input signal" 
	     << out   << "output signal" 
	     << "Volt\n" ;

	// FFT calculation
	spectrum << time << in >> freq >> fftin << '\n';
	spectrum << time << out >> freq >> fftout >> '\n';

	// plot frequency domain signal
	plot << "FIR FILTER" 
             << freq  << "Hz"           << 1e6 >> 300e6 
	     << fftin << "FFT input signal"  << LOG
	     << fftout<< "FFT output signal"
	     << "Magnitude\n";
}

void fir(array& in,array& filter,array& out)
{
	int i,j,k;
	array fil=array(0.0,0.0 ,filter.n);
	double sum=0;
	k=fil.n;
	for(i=0;i<k;i++) sum += filter.dat[i];
	fil = filter/sum;
	out=conv(in,fil);
}

