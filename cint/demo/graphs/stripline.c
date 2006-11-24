/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
* stripline impedance & propagation delay calculation
*
* array pre-compiled class library is needed to run 
* this demo program.
*************************************************************************/
#include <array.h>

main()
{
	microstripline(35e-6,5.0);
	stripline(35e-6,5.0);
	// twistedpair();
	// coaxial();
}

//  MICRO STRIP LINE
//
//                       |<--  W  -->|
//                  _
//                T _    /////////////
//          -----------------------------------------
//              ^
//              H                     ER
//              v
//          -----------------------------------------
//          /////////////////////////////////////////
//
//
//      Zo=87/sqr(ER+1.41)*ln(5.98*H/(.8*W+T))   ohm
//
//      Tpd=3.34*sqr(.475*ER+.67)    ns/m
//
//      Zo=sqr(Lo/Co)
//
microstripline(
	       double T=35e-6  // metal thickness
	       ,double ER=5.0
	       )
{
	int i;
	double H[6]={ 0.1e-3 , 0.2e-3 , 0.3e-3 , 0.5e-3, 0.7e-3, 1.0e-3 };
	char *name[6] = { "H=0.1mm" , "0.2mm" , "0.3mm" 
			  ,"0.5mm","0.7mm" ,"1.0mm"};
	char title[100];
	array W=array(0.01e-3,2e-3,100);
	array Zom[6];

	sprintf(title,"Microstripline Z(Er=%g,T=%g)",ER,T);

	for(i=0;i<6;i++) {
		Zom[i]=87/sqrt(ER+1.41)*log(5.98*H[i]/(0.8*W+T));
	}

	plot << title << W << "Width" ;
	for(i=0;i<6;i++) {
		plot << Zom[i] << name[i] ;
	}
	plot << 0 >> 150 << "ohm\n" ;

	// cout << "Micro stripline Tpd=" << 3.34*sqrt(0.475*ER+0.67) << "\n";
}

// STRIP LINE
//
//          /////////////////////////////////////////
//          -----------------------------------------
//          ^
//          |            |<--  W  -->|
//          |       _
//          B     T _    ///////////// |    
//          |                                 ER
//          v    
//          -----------------------------------------
//          /////////////////////////////////////////
//
//          Zo=60/sqr(ER)*ln(4*B/.67/3.14/W/(.8+T/W))  ohm
//
//          Tpd=3.34*SQR(ER)    ns/m
//
stripline(
	  double T=35e-6  // metal thickness
	  ,double ER=5.0
	  )
{
	int i;
	double B[]={ 0.1e-3 , 0.2e-3 , 0.3e-3 , 0.5e-3, 0.7e-3, 1.0e-3 };
	char *name[] = { "B=0.1mm" , "0.2mm" , "0.3mm" 
			  ,"0.5mm","0.7mm" ,"1.0mm"};
	char title[100];
	array W=array(0.01e-3,2e-3,100);
	array Zo[6];

	sprintf(title,"Stripline Z(Er=%g,T=%g)",ER,T);

	for(i=0;i<6;i++) {
		Zo[i]=60/sqrt(ER)*log(4*B[i]/.67/3.14/W/(.8+T/W))  ;
	}

	plot << title << W << "Width" ;
	for(i=0;i<6;i++) {
		plot << Zo[i] << name[i] ;
	}
	plot << 0 >> 80 << "ohm\n";

	// cout << "Stripline Tpd=" << 3.34*sqrt(ER) << "\n";
}


// TWISTED PAIR CABLE
//
//         DIAMETER=2*R
//         DISTANCE=D
//
//         If  D/(2*R) >> 1 then
//
//           C=PI*ER*Eo/ln(D/R)
//           L=Uo*UR/PI*ln(D/R)
//           Z=sqr(L/C)
//
twistedpair(
	    double radius
	    ,double distance
	    ,double ER=1.3
	    )
{
}


// COAXIAL CABLE
//
//           OUTER INSIDE DIAMETER = 2*B 
//           INNER OUTSIDE DIAMETER = 2*A
//           Zc=SQR(U/E)/2/PI*ln(B/A)
//
coaxial(
	double outerinsize
	,double inneroutside
	)
{
}
