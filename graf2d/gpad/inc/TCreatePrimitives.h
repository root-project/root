// @(#)root/gpad:$Id: TCreatePrimitives.h

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCreatePrimitives
#define ROOT_TCreatePrimitives

#include "TPad.h"

class TCreatePrimitives {

private:

   static void ExecuteTwoPoint(TPad &pad, Int_t mode, Int_t event, Int_t px, Int_t py);
   static void ExecuteText(TPad &pad, Int_t mode, Int_t event, Int_t px, Int_t py);
   static void ExecutePolyLine(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py);

public:

   TCreatePrimitives() = default;
   virtual ~TCreatePrimitives() = default;

   static void R__DEPRECATED(7,00, "Will be removed from public API") Ellipse(Int_t event, Int_t px, Int_t py,Int_t mode);
   static void R__DEPRECATED(7,00, "Will be removed from public API") Line(Int_t event, Int_t px, Int_t py, Int_t mode);
   static void R__DEPRECATED(7,00, "Will be removed from public API") Pad(Int_t event, Int_t px, Int_t py, Int_t);
   static void R__DEPRECATED(7,00, "Will be removed from public API") Pave(Int_t event, Int_t px, Int_t py, Int_t mode);
   static void R__DEPRECATED(7,00, "Will be removed from public API") PolyLine(Int_t event, Int_t px, Int_t py, Int_t mode);
   static void R__DEPRECATED(7,00, "Will be removed from public API") Text(Int_t event, Int_t px, Int_t py, Int_t mode);

   static void ExecuteEvent(TPad &pad, Int_t edit_mode, Int_t event, Int_t px, Int_t py);
};

#endif
