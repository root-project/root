/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
* modulation.c
*
* array and fft pre-compiled class library is needed to run 
* this demo program.
*************************************************************************/
#include <array.h>
#include <fft.h>

const double PI=3.141592;

main() {
	// create AM waveform and plot
	               // start , stop  ,npoint
	array time=array(-20*PI , 20*PI , 256 ),y1; 
	y1 = sin(time)*sin(time/10);
	plot << "time domain" << time << "time" << y1 << "\n";

	// FFT and save data
	array freq,y2;
	spectrum << time << y1 >> freq >> y2 >> endl;
	arrayostream("datafile") << "freq domain" << freq << y2 << endl;

	// load data and and plot
	graphbuf buf;
	arrayistream("datafile") >> buf;
	plot << buf ;
}
