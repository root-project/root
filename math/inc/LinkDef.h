/* @(#)root/math:$Name:  $:$Id: LinkDef.h,v 1.18 2007/02/05 18:07:10 brun Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link C++ nestedtypedefs;
#pragma link C++ nestedclasses;


// from base/inc/LinkDef1.h
#pragma link C++ global gRandom;

// from base/inc/LinkDef2.h
#pragma link C++ namespace TMath;

#pragma link C++ function operator*(Double_t, const TComplex&);
#pragma link C++ function operator+(Double_t, const TComplex&);
#pragma link C++ function operator/(Double_t, const TComplex&);
#pragma link C++ function operator-(Double_t, const TComplex&);
#pragma link C++ function operator>>(istream&,TComplex&);
#pragma link C++ function operator<<(ostream&,const TComplex&);

#pragma link C++ class TComplex+;
#pragma link C++ class TRandom+;
#pragma link C++ class TRandom1+;
#pragma link C++ class TRandom2+;
#pragma link C++ class TRandom3-;

#endif
