/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/******************************************************
* fir2.c
*
*  FIR filter simulation2
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
        // FIR transient response
	filter = sinc(filter); 

	array time=array(0.0 ,1e-6 ,512); //start,stop,datapoint
	array in;
	array out;
	array freq;
	carray in_fft;
	array out_re,out_im;


	// input signal calculation
	in = sin(w1*time)*0.4+sin(w2*time);

	// FIR filter
	fir(in,filter,out);

	// plot time domain signal
	plot << "FIR filter" 
             << time  << "time"
	     << in << "input signal" 
	     << out << "output signal" 
	     << "\n" ;

	// FFT calculation
	fft << time << in >> freq >> in_fft << '\n';
	fft << time << out >> freq >> out_re >> out_im << '\n';

	// plot frequency domain signal
	plot 	<< "FIR FILTER" 
                << freq << "freq" << 0 >> 100e6
	        << in_fft << "FFT input signal"
		<< out_re << "FFT output signal(re)"
		<< out_im << "FFT output signal(im)"
		<< "\n";

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

