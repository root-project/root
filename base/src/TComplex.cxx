// @(#)root/base:$Name:  $:$Id: TComplex.cxx,v 1.3 2005/12/24 17:06:27 rdm Exp $
// Author: Federico Carminati   22/04/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TComplex                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TComplex.h"
#include "Riostream.h"


ClassImp(TComplex)

//______________________________________________________________________________
TComplex::TComplex(Double_t re, Double_t im, Bool_t polar) : fRe(re), fIm(im)
{
   // Standard constructor

   if (polar) {
      fRe=re*std::cos(im);
      fIm=re*std::sin(im);
   }
}

//______________________________________________________________________________
ostream& operator<<(ostream& out, const TComplex& c)
{
   out << "(" << c.fRe << "," << c.fIm << "i)";
   return out;
}

//______________________________________________________________________________
istream& operator>>(istream& in, TComplex& c)
{
   in >> c.fRe >> c.fIm;
   return in;
}
