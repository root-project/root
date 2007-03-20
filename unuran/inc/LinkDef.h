/* @(#)root/minuit:$Name:  $:$Id: LinkDef.h,v 1.2 2007/03/08 09:31:54 moneta Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


#pragma link C++ class TUnuran+;
#pragma link C++ class TUnuranContDist+;
#pragma link C++ class TUnuranMultiContDist+;
#pragma link C++ class TUnuranDiscrDist+;
#pragma link C++ class TUnuranEmpDist+;

#pragma link C++ function  TUnuranDiscrDist::TUnuranDiscrDist (double *, double*);
#pragma link C++ function  TUnuranEmpDist::TUnuranEmpDist (double *, double*,unsigned int);


#endif
