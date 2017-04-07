// @(#)root/g3d:$Id$
// Author: Rene Brun   26/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCTUB
#define ROOT_TCTUB


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TCTUB                                                                  //
//                                                                        //
// 'CTUB' is a cut  tube with 11 parameters.  The  first 5 parameters     //
//        are the same  as for the TUBS.  The  remaining 6 parameters     //
//        are the director  cosines of the surfaces  cutting the tube     //
//        respectively at the low and high Z values.                      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TTUBS.h"

class TCTUB : public TTUBS {

protected:
   Float_t fCosLow[3];        // dir cosinus of surface cutting tube at low z
   Float_t fCosHigh[3];       // dir cosinus of surface cutting tube at high z

   virtual void    SetPoints(Double_t *points) const;
public:
   TCTUB();
   TCTUB(const char *name, const char *title, const char *material, Float_t rmin,
         Float_t rmax, Float_t dz, Float_t phi1, Float_t phi2,
         Float_t coslx, Float_t cosly, Float_t coslz,
         Float_t coshx, Float_t coshy, Float_t coshz);
   TCTUB(const char *name, const char *title, const char *material, Float_t rmin,
         Float_t rmax, Float_t dz, Float_t phi1, Float_t phi2,
         Float_t *lowNormal, Float_t *highNormal);
   virtual ~TCTUB();

   ClassDef(TCTUB,2)  //The Cut Tube shape
};

#endif
