/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**********************************************************************
* jitter.c
*
* array pre-compiled class library is needed to run 
* this demo program.
**********************************************************************/
#include <array.h>

const double cl=0.5e-12;
const double K=1.38e-23;  
const double T=300;
const double Vdd=5.0;
const double Vt=1.1;
const double un=0.048;
const double eox=3.45e-11;
const double tox=200e-10;
const double Kd=un*eox/tox;
const double Kf=3e-21;
const double PI=3.141592;

main()
{
	int i;
	array W=array(2e-6,30e-6, 100);
	array Id,gm;
	array Tj1[3],Tj2[3],Tj[3];
	double L[]= { 1e-6 , 2e-6 , 3e-6 };
	char *dataname[] = { "L=1um", "2um" , "3um" };

	plot << "CMOS inverter jitter" << W << "Width" ;

	for(i=0;i<3;i++) {
		// Johnson noise
		Tj1[i]=sqrt(2*K*T*cl/PI)*2*L[i]/Kd/W/(Vdd-Vt)@2;

		// Flicker noise
		gm=Kd*W/L[i]*(Vdd-Vt);
		Id=Kd*W/2/L[i]*(Vdd-Vt)@2;
		Tj2[i]=cl*Vdd/Id@2 * gm *sqrt(Kf/L[i]/W*log(Id/cl/Vdd));

		// Total
		Tj[i]=sqrt(Tj1[i]@2+Tj2[i]@2);

		plot << Tj[i] << dataname[i] ;
	}

	plot << "sec\n";
}

