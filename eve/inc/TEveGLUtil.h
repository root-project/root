// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef Reve_TEveGLUtil
#define Reve_TEveGLUtil

#include "Rtypes.h"

class TAttMarker;
class TAttLine;

class TEveGLUtil
{
public:
   virtual ~TEveGLUtil() {}

   // Commonly used rendering primitives.

   static void RenderLine(const TAttLine& al, Float_t* p, Int_t n,
                          Bool_t selection=kFALSE, Bool_t sec_selection=kFALSE);

   static void RenderPolyMarkers(const TAttMarker& marker, Float_t* p, Int_t n,
                                 Bool_t selection=kFALSE, Bool_t sec_selection=kFALSE);

   static void RenderPoints(const TAttMarker& marker, Float_t* p, Int_t n,
                            Bool_t selection=kFALSE, Bool_t sec_selection=kFALSE);

   static void RenderCrosses(const TAttMarker& marker, Float_t* p, Int_t n, Bool_t sec_selection=kFALSE);

   ClassDef(TEveGLUtil, 0); // Commonly used utilities for GL rendering.
};

#endif
